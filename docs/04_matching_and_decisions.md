# 04 — Matching models and precision-aware decisions

## 1. Tonight's primary model

Train a compact gradient-boosted binary classifier over candidate-pair features. LightGBM is the default to benchmark; its official repository declares an MIT license. Use the installed, verified version and record it. This is a design choice, not a claim that it wins on the supplied data. [Official license](https://github.com/lightgbm-org/LightGBM/blob/main/LICENSE)

Keep a high-precision deterministic composite-match baseline as the first end-to-end control. It must score and export its candidates honestly. Never accept exact-name matches automatically when addresses disagree or names are common.

### Feature families

| Family | Proposed features | Identity signal |
| --- | --- | --- |
| Name | Raw/normalized exact match, edit similarity, token overlap, weighted Jaccard, character cosine, token-order similarity | Spelling and order variation |
| Name specificity | Rare-token agreement, frequency, acronym agreement, length ratio, common-name flag | Shared generic names are weak evidence |
| Address | Token/character overlap, containment, edit similarity, component presence | Incomplete address agreement |
| Numeric evidence | Number-set overlap, possible house/unit/postal agreement and conflict, parser confidence | Branch and location discrimination |
| Missingness | Empty fields, partial address, short names, unmatched components | Missing evidence is not a contradiction |
| Retrieval | Channel membership, rank, score, number of agreeing channels | Retrieval reliability |
| Candidate context | Candidate count, top score, gaps, competing anchors if measured | Ambiguity and singleton cues |
| Source/country relation | S2 vs S3, country agreement/unknown flags | Source noise and country consistency |
| Optional embedding | Combined/name/address cosine from frozen encoder | Multilingual/low-lexical-overlap support |

Country equality is useful; a closed country one-hot encoding is not. Do not feed ID digits or row positions into the model. Separate missing values from zeros and explicit conflicts. Learn or validate conflict effects rather than using an untested universal postal-code veto.

Use vectorized/native implementations for expensive edit features only on bounded candidates. Benchmark feature generation: it can dominate runtime even when the classifier is fast.

## 2. Training examples and sampling

Generate fit-side candidates using the same blocking recipe as inference. Label candidates positive when the target belongs to the supplied truth set and negative otherwise, subject to the integrity audit and split restrictions.

Include hard negatives: same/common name with different address, similar address with a distinct name, high lexical/dense retrieval scores, and plausible candidates for true singletons. Random negatives alone make an unrealistically easy training task.

For the first fit, use a stratified subset of complete S1 groups with all their candidate lists; expand if learning curves and time justify it. Initial training sample targets can be 25k–100k anchors, adjusted to measured data size and memory. Avoid allocating tens of millions of full-text pair objects.

If negative downsampling is necessary, retain hard negatives plus a recorded random sample, store sampling probabilities, and compare sampling/weighting strategies on untouched natural candidate distributions. Downsampling changes probability calibration. Do not interpret raw classifier output as a deployment posterior by default.

To reflect macro scoring, compare ordinary pair loss with per-anchor normalized sample weights so large candidate lists do not dominate. Tune class weights empirically; overweighting positives can damage the precision-heavy metric. The final choice is made by end-to-end macro F₀.₅, not training loss.

Retrieved training positives are the baseline. Optionally inject missed training positives to improve representations, tagged explicitly; never count those injected pairs as successful blocking and never inject validation/test truth.

## 3. Training sequence

1. Fit a small tree model on a stratified group sample; early-stop using development pair loss and inspect end-to-end F₀.₅.
2. Tune only a small set of capacity/regularization choices, not a large search: tree depth/leaves, minimum leaf population, learning rate, and negative mixture.
3. Improve the most common false-merge/missed-match feature deficiency.
4. Expand training data when the model benefits and the run remains within budget.
5. Add frozen multilingual features only if already-computed embeddings justify their cost.

Save exact features/order, preprocessing state, training IDs, seeds, model bytes, evaluation predictions, and the chosen iteration count. The first reproducible useful model is the release fallback.

## 4. Set prediction and threshold selection

For the first model, select every candidate whose score exceeds a development-tuned global threshold. Return an empty list if none qualify. Search observed score breakpoints or a sufficiently fine grid against the exact macro metric, not a fixed 0.5 rule.

Multiple confident targets are valid. Do not impose a large top-one margin that rejects genuine multi-match groups, and do not force a fixed number of outputs. Candidate top-k and final match count are different decisions.

Progressive optional decision rules:

- Source-specific thresholds if both sources have enough development support and gains are stable.
- Confidence/missingness-dependent abstention using simple validated rules.
- A small S1-level singleton gate trained from out-of-fold candidate evidence: top scores, score distribution, independent field agreement, ambiguity, and missingness.

A singleton gate cannot recover candidates that were never generated. Evaluate missed non-singletons as well as reduced false merges. Keep the simpler global rule if the gate overfits or offers negligible improvement.

If calibrating probabilities, fit a calibrator on a disjoint calibration partition or out-of-fold scores generated without those labels; tune its decision rule on separate development evidence. Prefer a simple calibration model when support is limited. Calibration is optional for thresholding, and should not delay the baseline.

Choose thresholds by full predicted sets. The formula's `4*FP` term does not imply a universal optimal threshold because scores, priors, match multiplicity, singleton credit, and macro aggregation all matter.

## 5. France and unseen-country behavior

- Keep character and token evidence as a language-agnostic foundation.
- Preserve accents and accent-folded variants together, rather than requiring either exact accented text or English-only normalization.
- Use generic numeric/component evidence when country-specific parsing is unavailable.
- Share the global decision rule for unseen countries; no France-specific threshold chosen without French labels.
- Evaluate shift robustness with leave-country-out experiments where feasible.
- If a multilingual encoder is added, verify it improves hard cases and does not merely match semantic business categories.

Do not manufacture confidence by returning empty predictions for all France entities. Include every France S1 and apply the same validated generic pipeline, with explicit missing-evidence handling.

## 6. Neural options, ranked by practicality

### A. Frozen compact multilingual embeddings — conditional upgrade

Use the compact E5 channel described in the blocking plan. Reuse its cached similarities as features. This requires no fine-tuning and offers the lowest-risk GPU experiment, but full dataset encoding may still exceed the available time.

### B. Fine-tuned bi-encoder — stretch, deferred by default tonight

Only if the baseline exists and schedule permits: fine-tune eligible pretrained weights with positives from the training groups and mined hard negatives. Use a multi-positive contrastive objective; mask known matches so another positive for the same S1 is not treated as an in-batch negative. Include all known same-entity records when constructing those masks.

Augmentation is limited to train-only deterministic perturbations that preserve identity, such as punctuation changes or modest component dropout. Avoid number changes that turn a location into another business. Train no representation on holdout labels.

Monitor leave-country-out degradation. Fine-tuning on US/India can harm the original multilingual representation; compare against the frozen encoder before promoting it.

### C. Pair cross-encoder/reranker — stretch, deferred by default tonight

Serialize both records with explicit field labels and train/evaluate a binary identity scorer. Benchmark at 128/256 total tokens with field-aware truncation and dynamically padded batches. Preserve both names and important address evidence when truncating.

`BAAI/bge-reranker-v2-m3` is a multilingual candidate whose card declares Apache-2.0. Its generic relevance score is not a calibrated identity probability. Consider it only after hardware/license/count checks and a measured cost-benefit experiment. [Official model card](https://huggingface.co/BAAI/bge-reranker-v2-m3)

If the tabular model scores all candidates and the neural model only rescans uncertain pairs, log that routing and export all tabular-scored candidates. Train/evaluate the same cascade, including routing errors and boundary cases. Changing the reranker budget at inference changes the system and requires validation.

### D. Larger dense encoder — deferred option

`BAAI/bge-m3` declares MIT and multilingual retrieval support. It is an alternative representation experiment, not required for tonight's submission. The compact encoder is the first option for the unspecified laptop GPU. [Official model card](https://huggingface.co/BAAI/bge-m3)

For every neural model, verify exact parameter count and license of the pinned revision before adoption. Conservatively keep the total deployed neural parameters below 8B, and include the necessary notices. A model-card license label is a starting check, not a substitute for preserving its actual license files.

## 7. Ensembling and global consistency

Only ensemble models with complementary held-out errors. Fit simple blend weights on development or out-of-fold scores and evaluate against the best single model. Include the additional candidate set, memory, runtime, and packaging cost in the decision.

Source 1 is deduplicated, but that alone does not justify Hungarian one-to-one assignment: each S1 can match many targets. If the label audit supports unique target ownership, a conservative conflict resolver may assign a target to the strongest anchor only with a validated confidence margin; otherwise abstain or retain the baseline behavior. Evaluate it end to end.

No unrestricted transitive closure or connected-component match expansion. A chain of plausible edges can create a false merge. Any proposed new S1-target edge must be generated, scored, and included in the candidate audit.

## 8. Promotion conditions

Keep an upgrade only when it offers reproducible end-to-end improvement or similar quality at materially lower candidate/runtime cost, protects singleton and country-shift behavior, completes before the freeze, and passes all output/provenance checks. No model is promoted based solely on training metrics, semantic similarity demos, or a public leaderboard fluctuation.
