# 02 — Data audit and validation design

All steps in this document execute on the compute PC. The statistics below are requested outputs, not findings already observed.

## 1. First remote data audit

Read the TSVs in streaming batches with an explicit tab separator and string columns. Preserve literal business text and empty strings; avoid automatic NA conversion. Keep raw files immutable and store SHA-256 hashes, byte sizes, schema, and measured row counts in a data manifest.

Validate:

- Required column names; row widths; UTF-8 decoding; embedded tabs/newlines and quoting behavior.
- Unique entity IDs within each source; correct prefixes; no accidental train/test ID collisions.
- Exactly one ground-truth row per training S1, no unknown S1, duplicate match IDs, or S1 targets.
- Every labeled target exists in training S2/S3; every ground-truth target's country relationship is recorded.
- Target reuse across multiple S1 anchors. If present, group the connected anchors together for splits and investigate; do not silently impose exclusive ownership.
- Missing/empty names, addresses, countries; country labels and variants; text-length and Unicode/script distributions.
- Duplicate raw and normalized name/address records, cross-split text overlap, common-name collisions, large posting-list sizes.

Profile truth multiplicity overall and by country/source: singleton fraction, 1/2/3+ matches, high quantiles, maximum, and S2/S3 distribution. Measure known positives' name/address similarity and disagreements. Country mismatch determines whether hard country blocking is safe.

Export compact aggregate reports, not a giant notebook output. Audit unusual records locally on the compute PC without internet lookup. Test records may be summarized for input coverage and schema drift, but must not supply labels, hand-edited matches, or optimization feedback.

## 2. Validation units and split recipe

**Never randomly split labeled pairs.** An S1 and its positive records must not appear in both supervised training and validation.

Proposed initial split is **70% fit / 15% development / 15% locked holdout**, assigned by stable group hash with seed 42. Exact fractions can change before the manifest is frozen if counts are unexpectedly small. Stratify groups approximately by country, singleton status, multiplicity, and source mix.

Construct groups from the full labeled positive graph: S1 anchors and their S2/S3 positives belong to one connected component. An ordinary component will be one S1 plus its matches; shared targets expand it. Reserve all labeled target IDs in development/holdout components from supervised fitting, including from negative-mining pools. Quarantine unlinked exact text duplicates of held-out records from supervised fitting if the audit identifies them.

For the baseline, supervised fitting and negative mining use fit-side S1 and the allowed fit target pool. Model selection and thresholding use development S1 against a realistic full training S2/S3 retrieval index. The locked holdout likewise queries the full target pool, so common-name distractors and index scale remain realistic. Merely indexing held-out target text for evaluation is necessary inference access, not permission to train on it. Fit learned preprocessing, vocabulary/IDF, pruning models, and normalizers on the fit partition; transform validation records with frozen artifacts.

Track unlinked S2/S3 records separately: they remain important distractors. Treat unlabeled pairs as negatives only under the challenge's exhaustive-match-label interpretation, and review apparent contradictory examples rather than relabeling by guesswork.

Save exact group/partition membership, configuration, hashes, and split version. Audit supervised training pair endpoints against reserved IDs before every fit. Do not split by contiguous ID ranges or take the first N rows as a representative sample.

**Deadline adaptation:** freeze the full partition manifest once, then use a fixed stratified 5k–10k-anchor subset of development for rapid iterations, scaled down if the data is smaller. Query it against the full realistic target index. Expand development evaluation for the selected release when runtime permits; report the exact evaluated population and do not present subset metrics as full-development results. Keep the locked holdout separate throughout. Group cross-validation and bootstrap analysis are secondary to finishing the baseline today.

### Stronger stress tests

1. **Leave-country-out:** fit on US, evaluate on India; reverse direction. Inside the training country, retain a development portion for thresholds so held-out-country labels do not tune that model.
2. **Name-family stress split:** group similar normalized business-name families before splitting, then inspect common chain names and new-name generalization. Do not union every record sharing a common token; avoid one giant component.
3. **Noise slices:** naturally missing address/postal data, low lexical overlap, short/common names, conflicting numeric components, Unicode and accent variants.
4. **Optional robustness perturbations:** controlled removal of address components or punctuation on validation copies. Keep these separate from the untouched main metric.

Leave-country-out is a proxy for distribution shift, **not a France validation score**. France has no supplied labels, so real French accuracy cannot be established locally.

## 3. Exact metric implementation

For each S1 entity `i`, let `G_i` be the true target-ID set and `P_i` the predicted set.

```text
TP_i = |P_i ∩ G_i|
FP_i = |P_i \ G_i|
FN_i = |G_i \ P_i|

If G_i is empty:
    F_i = 1 if P_i is empty, else 0
Otherwise:
    F_i = 5*TP_i / (5*TP_i + 4*FP_i + FN_i)

macro_F0.5 = sum(F_i) / number_of_S1_entities
```

This is equivalent to the provided precision/recall formula with β=0.5. For a non-singleton, an empty prediction gives zero. Every S1 participates, including those with zero candidates. Never average only over entities that had a retrieved pair or predicted match.

Required hand-checked scorer cases:

| Truth | Prediction | Expected per-entity score |
| --- | --- | ---: |
| Empty | Empty | 1 |
| Empty | `{a}` | 0 |
| `{a}` | Empty | 0 |
| `{a}` | `{a}` | 1 |
| `{a}` | `{b}` | 0 |
| `{a,b}` | `{a}` | 5/6 |
| `{a,b}` | `{a,b,c}` | 5/7 |

Also test macro versus micro aggregation, multiple matches in one source, duplicate input handling, and complete S1 coverage. Invalid output should fail validation before scoring rather than silently benefit from set deduplication.

An all-empty baseline scores exactly the singleton fraction. An all-empty candidate file does not make this a competitive system; retain it only as a metric sanity check.

## 4. Retrieval metrics and achievable score

Let `C_i` be the final candidate set and `r_i = |C_i ∩ G_i|`.

- Micro pair recall: `sum(r_i) / sum(|G_i|)`.
- Macro candidate recall: average `r_i / |G_i|` over non-singletons.
- Full-recovery rate: proportion of non-singletons with every true match in `C_i`.
- At-least-one recovery: proportion of non-singletons with `r_i > 0`.
- Candidate count: mean, median, p95, p99, maximum, and total; separate countries and S2/S3.
- Reduction ratio: `1 - sum(|C_i|)/(N1*(N2+N3))`; also report the country-partition denominator explicitly when used.
- Empty-candidate rate for singletons and non-singletons separately.

The **oracle macro F₀.₅ ceiling** predicts only the true positives present in the candidate set. It scores 1 for true singletons and `5*r_i/(4*r_i + |G_i|)` for non-singletons. This cleanly isolates blocking loss. High micro recall can hide failures on many low-multiplicity entities; the macro ceiling is essential.

Report retrieval-only metrics before reporting classifier performance. Never insert validation positives into candidates to improve an end-to-end result. Any training-only gold-positive injection must be flagged and excluded from retrieval metrics.

## 5. Model evaluation protocol

Measure end-to-end macro F₀.₅ on natural candidate distributions, plus micro precision/recall as diagnostics. Report singleton false-positive rate, matched-entity recall, mean prediction count, exact-set accuracy, and source/country/multiplicity slices. Pair accuracy and pair AUC do not select the final system.

For differences between systems, use paired bootstrap intervals over S1 groups and inspect the per-entity delta distribution. Record changed false merges and recovered matches. Small nominal gains that disappear under resampling, or substantially harm a country-shift slice, do not justify a more expensive pipeline.

Use development data for blocker budgets, model hyperparameters, thresholds, and ensemble weights. Keep the holdout untouched until selecting a release candidate; repeated examination turns it into development data and must be acknowledged. If time allows, use 3-fold group out-of-fold predictions inside the non-holdout portion for calibration and blending. Every learned stage used to generate those predictions must exclude that fold's labels.

After selection, freeze the recipe. Refit on all eligible training labels for test inference with the validated training schedule; retain the out-of-fold/development decision rule. Audit score-scale drift after refitting. If calibration changes materially, prefer the validated model/fold ensemble or another validated calibration procedure rather than choosing a threshold from test outputs.

## 6. Error-analysis ledger

Each reviewed development error records run, S1, pair IDs, truth, prediction, candidate provenance, feature evidence, and one cause: missing candidate, wrong merge, missed scored positive, singleton error, source conflict, parser problem, or suspected label ambiguity.

For every proposed fix, state the expected affected slice and rerun the complete held-out candidate pipeline. Prioritize high-frequency causes and inspect both gains and newly introduced false positives. Keep example records in local artifacts; publish only aggregate summaries and approved synthetic illustrations in Git.
