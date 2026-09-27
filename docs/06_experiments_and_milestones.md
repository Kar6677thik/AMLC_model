# 06 — Deadline execution plan and experiments

Team: **RestoreBuildRun**. User deadline: **tonight, 2026-09-27, 23:59**. Times below assume **IST (Asia/Kolkata)**; confirm the portal timezone. Clock check during planning was approximately **08:57 IST**, so this is a same-day plan with about 15 hours available at that point, not a multi-day research agenda.

## 1. What we should commit to today

Deliver a reproducible lexical-blocking/tabular-matching system, both valid output files, runnable packaged code, and a truthful methodology document. Start full test inference early. An excellent GPU does not eliminate CPU indexing, feature extraction, or packaging bottlenecks.

The optional upgrade is frozen compact multilingual retrieval/features. Fine-tuned bi-encoders, large cross-encoders, sophisticated graph reconciliation, and broad ensemble searches are **deferred by default**. A finished validated baseline is the required checkpoint before any of those experiments.

## 2. Proposed same-day schedule

The windows are budgets and stop points, not runtime predictions. Move expensive stages earlier if the benchmark demands it; reduce optional scope if implementation starts later.

| IST window | Work | Concrete exit artifact |
| --- | --- | --- |
| 09:00–09:45 | Hardware/environment preflight, safe repo setup, data transfer/hash/audit; begin scorer and I/O | Hardware/data manifests; verified input contract |
| 09:45–11:30 | Normalization, simple indexed channels, metric fixtures, group split, deterministic baseline | Small complete run producing both TSVs |
| 11:30–13:00 | Pair features, sampled tree fit, threshold tuning, realistic throughput benchmark | Reproducible development report; full inference ETA |
| **By 13:00** | Start full baseline test pipeline, or earlier if ETA requires | Resumable full inference in progress |
| 13:00–16:30 | Inspect baseline errors; at most 1–2 targeted improvements; optional compact GPU pilot | Compared candidate/score/cost reports; preserve baseline |
| 16:30–18:00 | Select final configuration; evaluate locked holdout once; decide validated model vs refit | Frozen release configuration and model |
| **18:00 onward** | No new architectures; finish selected full inference and output checks | Two final TSVs and provenance ledger |
| **Target by 21:00** | Validate, upload a verified matching file, inspect portal response; assemble/reproduce zip | Accepted/scored upload where portal permits; release package |
| 21:00–22:30 | Resolve rejection/format/runtime issues and finalize methodology/zip | Final accepted artifacts with hashes |
| **22:30–23:59** | Protected contingency buffer; avoid risky last-minute model changes | Confirmation/receipt and archived final release |

If portal submissions permit, upload the first validated full baseline as soon as it is ready, before the target final upload. Respect the actual submission quota; do not assume unlimited uploads. The live leaderboard accepts `matching_results.tsv`; the final package additionally needs candidates, code, and documentation.

If a full run's conservative ETA extends into the contingency window, immediately choose the already-validated cheaper recipe. Do not reduce inference candidate budgets or thresholds without evaluating the changed recipe. If necessary, use the known-good deterministic baseline and document its limits.

## 3. Milestones with acceptance criteria

### M0 — Correctness foundations

Implement input validation, string-preserving TSV loading/writing, complete S1 coverage, metric fixtures, and split manifest. Audit country labels and label references. Configure remote runtime and source synchronization.

Done when: scorer matches the hand-computed cases; a synthetic multi-match/singleton/unseen-country fixture round-trips; no data/model artifacts are accidentally staged in Git; data is readable on the compute PC.

### M1 — First complete baseline

Implement conservative normalization, compact indexed blocking, deterministic pair scoring, and both exports. Run sampled queries against realistic distractor pools; measure oracle ceiling and candidates.

Done when: two TSVs pass internal strict checks and the organizer validator; scored candidates exactly equal exported candidates; stage time and memory are recorded. This is an engineering checkpoint, not evidence of strong model quality.

### M2 — Useful supervised baseline

Generate fit-side pair features, train a small LightGBM classifier, tune a global threshold on development, and measure singleton/multi-match behavior. Benchmark full-size index querying and inference; launch the full baseline early.

Done when: end-to-end development score is compared against all-empty and deterministic controls, retrieval loss is separated from classifier loss, and full inference has a conservative completion estimate.

### M3 — Targeted improvement

Fix the largest measured failure: retrieval misses → another bounded lexical view; branch false merges → numeric/address specificity; singleton false merges → stronger evidence/threshold; low lexical overlap → compact multilingual pilot if feasible.

Done when: the change improves the score/candidate/runtime trade-off on the same development groups and does not introduce a major slice regression. Otherwise revert to the previous complete version.

### M4 — Release

Freeze the recipe, assess locked holdout once, finish full inference, validate provenance/IDs/coverage, reproduce from packaged contents, complete methodology, and produce `RestoreBuildRun_submission.zip`.

Done when: both files and package exist; their hashes agree with the run ledger; final uploaded matching file matches the packaged one; portal status/receipt is checked by the submitting team member.

## 4. Experiment matrix

Each run changes one meaningful factor. No large hyperparameter sweep today.

| ID | Experiment | Question | Priority |
| --- | --- | --- | --- |
| E00 | All-empty scorer control | Does score equal singleton fraction? | Required |
| E01 | Composite/rare-token blocking + deterministic evidence | Can the complete pipeline run and validate? | Required |
| E02 | Lexical candidates + tabular classifier + global threshold | Is supervised matching useful over controls? | Required |
| E03 | Candidate caps/channels on development | What is the smallest useful candidate set? | Required, small grid |
| E04 | Numeric conflicts, rarity, missingness | Can we reduce the dominant false merges? | High if supported by errors |
| E05 | Frozen compact multilingual channel | Does it recover lexical misses within runtime budget? | Conditional |
| E06 | Frozen embedding similarity as pair feature | Does retrieved semantic evidence help identity decisions? | Conditional on E05 artifacts |
| E07 | Source threshold/singleton rule | Can simple decisions improve macro F₀.₅ reliably? | Conditional |
| E08 | Leave-country-out small-model probe | How brittle is the approach under country shift? | Important diagnostic, bounded |
| E09 | Fine-tuned retriever, cross-encoder, ensembles | Is added complexity justified? | Deferred by default |

For the ideal later research path, add group cross-validation, larger training samples, contrastive retrieval training, out-of-fold ensembles, and more extensive ANN tuning. Those are not prerequisites for tonight's release.

## 5. Minimal run report

```text
run_id / timestamp / owner
git_sha / input_manifest_hash / split_version / resolved_config_hash
training_anchor_count / candidate_pair_count / model_hash
retrieval_channels / candidate_budget / index_parameters
dev_macro_F0.5 / singleton_false_positive_rate / micro_precision / micro_recall
macro_oracle_ceiling / candidate_micro_recall / full_recovery_rate
mean_p95_p99_max_candidates / per_country_and_source_slices
stage_seconds / total_seconds / peak_RAM / peak_VRAM
decision_threshold / output_hashes / validator_result
change_tested / observed_tradeoff / keep_or_reject_reason
```

Use local JSON/CSV reports with small Markdown summaries. A hosted experiment tracker is optional and must not receive challenge rows. Do not spend time installing tracking infrastructure before the baseline.

## 6. Working roles if teammates are available

Assign one owner per interface: data/metric/splits; blocking/performance; matcher/decision rules; release/documentation. A small team can combine roles. Keep one shared split manifest and scorer so results remain comparable. Coordinate independent modules through Git, and avoid concurrent heavy jobs on the same laptop.

No teammate count has been provided and no sub-agents have been launched. This is an organizational suggestion, not a claim that tasks are already assigned.

## 7. Practical stop rules

- GPU unavailable or installation unstable: complete the CPU lexical/tabular route.
- Candidate recall poor: inspect missed positives before changing the classifier.
- Common-token posting lists dominate runtime: refine composite keys/caps and revalidate.
- Full feature matrix exceeds memory: train on complete sampled anchor groups; infer in shards.
- Neural pilot cannot finish encoding and inference with a safe margin: stop it and preserve the baseline.
- Validation gain is tiny or inconsistent: choose the simpler/cheaper system.
- France has no labels: do not invent country-specific performance or thresholds.
- New full run risks the final upload window: submit the already-validated release.
- Submission format or candidate provenance fails: fix correctness before further modeling.

## 8. Risk register

| Risk | Early signal | Mitigation |
| --- | --- | --- |
| Deadline missed | End-to-end ETA has no rerun/upload buffer | Start full baseline early; freeze by 18:00 |
| GPU assumptions wrong | No supported device/runtime | CPU baseline is independent of GPU |
| RAM exhaustion | Large Python string/pair objects; swapping | Compact indexes, shards, sampled fitting |
| Leakage | Shared anchors/positive targets across supervised splits | Graph groups and endpoint audits |
| France regression | Country-shift probe collapses | Generic lexical evidence; compact multilingual option |
| False merges on common names | High singleton FP or branch errors | Rarity, address/numeric conflicts, threshold tuning |
| Hidden blocker cost | Few outputs but huge posting visits | Log internal retrieval work and bound it |
| Candidate audit mismatch | Scored/exported pair hashes disagree | Export immutable matcher-input ledger |
| Public leaderboard overfit | Changes selected from tiny leaderboard movements | Development/holdout selection; limited uploads |
| Package cannot reproduce | Missing files, absolute paths, internet-only weights | Clean packaged-content smoke/reproduction run |

## 9. Immediate next implementation order

1. Confirm compute-PC access/environment and portal deadline timezone.
2. Add safe ignore rules and runnable package scaffold; synchronize code through the existing remote.
3. Implement audit, exact scorer, group split, and output validators.
4. Implement the smallest indexed lexical baseline and produce both files.
5. Add tabular matching, benchmark full inference, and launch the first complete test run.

This planning pass does not execute these steps. It makes the next coding session concrete and prioritizes the deadline-critical work.
