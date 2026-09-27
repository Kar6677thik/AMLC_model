# 01 — Requirements and decisions

Team: **RestoreBuildRun**  
Plan date: **2026-09-27**  
Status: proposed implementation; no experimental results yet.

## 1. Source of truth

Read together:

1. [Problem statement](../6ab5628d5a817_amazon_ml_challenge_problem_statement.md).
2. [Organizer resource README](../student_resource/README.md).
3. [Additional organizer guidance](../context1.md), which explicitly says smaller candidate sets affect final ranking beyond leaderboard scoring.
4. [Provided validator](../student_resource/utils/validate_submission.py).
5. [Required methodology template](../student_resource/Documentation_template.md).

The linked explanatory video has no usable URL in the supplied Markdown and was not reviewed. The plan does not assume facts from it. If organizer clarifications arrive, record their date and update this requirements ledger.

## 2. What we are solving

Source 1 is a deduplicated reference. For every Source 1 record, find **all** corresponding Source 2 and Source 3 record IDs. There may be no matches, one match, or multiple matches within either source. This is entity identity resolution from noisy business names, addresses, and country labels.

It is not a one-nearest-neighbor task. Do not force one result per source, merge records merely because they are in the same business category, or assume a global one-to-one assignment. Source 2/3 records with similar names can represent different branches or unrelated businesses.

## 3. Requirements ledger

| Requirement | Implementation consequence | Release evidence |
| --- | --- | --- |
| All input/output is TSV | Explicit tab delimiter, UTF-8, IDs as strings | Schema and round-trip checks |
| Exactly one output row for every test S1 | Left join predictions onto complete S1 manifest | Exact S1 set equality and row-count assertion |
| Empty results are valid | Preserve empty strings; never serialize `NaN`, `None`, or `[]` | Singleton/empty serialization fixtures |
| Many matches are allowed | Predict sets with independent pair evidence | Multi-match fixtures and multiplicity slices |
| Only existing test S2/S3 IDs | Resolve integer indices through the correct split/source map | Strict target membership checks |
| No duplicate rows or list IDs | Deterministic set serialization | Uniqueness checks |
| `candidate_pairs.tsv` is required in final zip | Persist immutable matcher input pairs | Candidate/scoring ledger equality |
| Matches must be candidates | Decision stage only selects from scored pairs | Hard subset assertion |
| Macro F₀.₅ includes singletons | Score each S1, then average | Hand-calculated scorer fixtures |
| France is unseen during training | Generic country handling and multilingual robustness | Leave-country-out tests; full France output coverage |
| Final model MIT/Apache-2.0; ≤8B parameters | License/revision ledger and parameter count | Licenses, model config, parameter manifest |
| External business lookup/augmentation forbidden | Only supplied records/labels enter task data | Data provenance and reproducible local pipeline |
| Candidate efficiency affects final ranking | Optimize recall versus candidate count and runtime | Candidate Pareto table and scaling report |
| Reproducible final zip | Include source, pinned environment, outputs, methodology | Clean-environment reproduction |

The organizers do not provide a numeric formula combining candidate efficiency with leaderboard F₀.₅. Do not invent one. Maintain a Pareto frontier and prefer materially smaller candidate sets when accuracy is practically equivalent.

## 4. Fair-play boundaries

Use the provided files as the only task-specific data. No geocoding, registries, search-engine business enrichment, entity resolution APIs, external address datasets, or downloaded lists of business identities. Do not send business records to online inference services.

The explicit model-license requirement and guidance about pretrained models support evaluating eligible pretrained checkpoints. Download their weights/tokenizers and ordinary software dependencies; run them locally. Public model documentation and licenses can inform engineering choices without becoming challenge data. Record every checkpoint's immutable revision, license, and provenance.

For the initial solution, use deterministic text normalization and optional train-only character/token perturbations. Avoid external French lexicons, pretrained address parsers with hidden external data assets, machine-translated augmentation, and LLM-generated business records. Any additional task-data source would require organizer clarification rather than an assumption.

No features from the numeric portions of entity IDs, file ordering, or apparent ID-generation patterns. IDs identify records and split membership; they are not business evidence.

## 5. Validator gaps and documentation inconsistencies

Inspection of `validate_submission.py` establishes that:

- Target-ID existence is **off by default**; use `--check-ids`.
- An absent candidate file produces a warning, even though it is mandatory for the final package.
- Matches outside the candidate set produce a warning rather than a failed run.
- The validator comments say nonexistent target IDs lower score; the written problem statement says they are rejected. We satisfy the stricter written rule and allow none.
- The validator validates formatting; it is **not** an official metric implementation.

Consequently, a validator PASS alone is insufficient. Our own release validation must hard-fail missing candidates, invalid IDs, incomplete coverage, and candidate/scorer mismatch.

The problem statement informally describes precision as weighted twice as heavily. Implement the actual formula: in count form its denominator contains `4 × FP` and `1 × FN`. Do not infer a universal probability threshold from that wording.

## 6. Observed local inventory

Only file metadata and organizer text/code were inspected; the datasets were not scanned.

| File | Bytes |
| --- | ---: |
| `train_source1.tsv` | 210,069,713 |
| `train_source2.tsv` | 489,301,488 |
| `train_source3.tsv` | 503,705,637 |
| `train_ground_truth.tsv` | 127,015,583 |
| `test_source1.tsv` | 175,022,086 |
| `test_source2.tsv` | 509,456,422 |
| `test_source3.tsv` | 506,002,772 |

The validator contains an approximate record-count comment. Treat it as documentation, not a measured property of these files. Row counts and memory estimates will be established remotely before choosing index sizes.

## 7. Architectural decisions

1. **Baseline first:** lexical blocking plus interpretable pair features and a gradient-boosted classifier.
2. **Separate retrieval from matching:** candidate misses and classification errors require different fixes.
3. **Multilingual upgrade:** add a compact multilingual encoder only after identifying retrieval failures.
4. **Precision-aware decisions:** tune complete predicted sets against macro F₀.₅, including empty predictions.
5. **Compute elsewhere:** this PC hosts planning/code edits; all data profiling, tests, training, benchmarking, and inference run on the compute PC.
6. **Reproducibility by design:** manifests, pinned configurations, immutable run directories, and exact output hashes.
7. **Small reliable final system:** additional models must earn their storage, inference, and reproduction cost.

## 8. Open information

| Missing information | Safe planning default | When it matters |
| --- | --- | --- |
| GPU and VRAM | GPU unspecified; baseline works without CUDA | Before model installation/benchmark |
| Exact CPU/GPU inventory and free disk | Confirmed Windows ThinkPad P16, Ultra 7, 32 GB RAM; streaming stages | Before data conversion/index build |
| Portal timezone and upload limits | User deadline tonight 23:59; assume 2026-09-27 IST pending portal check | Before time allocation and submissions |
| Team size/availability | Workstreams without named owners | Before task assignment |
| Repository visibility/access | Keep task data out of Git; inspect remote before first push | Before repository publishing |
| Whether pretrained checkpoints must be bundled | Include trained artifacts and pinned reconstruction path where permitted | Before packaging |
| Combined parameter cap for ensembles | Conservatively keep total deployed neural parameters under 8B | Before final model selection |
| Organizer handling of unresolved rule ambiguities | Use the stricter compliant behavior | Before depending on an ambiguity |

These unknowns do not block writing the plan or implementing the baseline. They do block truthful runtime promises and final hardware-specific environment pins.

## 9. Deadline update from the user

The user confirmed Windows, 32 GB RAM, an Ultra 7 CPU, a ThinkPad P16, and ample disk space. The exact GPU is unknown; a laptop described as powerful is not evidence of CUDA support or sufficient VRAM. The submission deadline is tonight at 23:59. Clock check: approximately 08:57 IST on 2026-09-27.

This makes the lexical/tabular path the committed plan. GPU-dependent retrieval is an optional upgrade, and fine-tuning/large rerankers are stretch work. Start full-size baseline inference early enough to measure its completion time. Reserve the last hours for reruns, integrity checks, package reproduction, and upload. No time-consuming environment migration is a prerequisite.
