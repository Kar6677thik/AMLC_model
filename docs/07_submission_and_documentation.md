# 07 — Submission, packaging, and methodology plan

## 1. Required deliverables

Team name: **RestoreBuildRun**. Final archive name:

```text
RestoreBuildRun_submission.zip
├── output/
│   ├── matching_results.tsv
│   └── candidate_pairs.tsv
├── code/
│   └── business_entity_resolution/
│       ├── src/
│       ├── README.md
│       ├── requirements.txt
│       ├── configs/                # final resolved settings
│       └── assets/                 # needed model/artifact assets, if used
└── Documentation_template.md
```

Required organizer folders and filenames must be exactly present at archive root, without an accidental extra outer directory. Additional reproducibility assets belong inside the runnable code folder. Do not include raw challenge data, transient caches, virtual environments, or unrelated files unless the organizer explicitly requires them.

Only `matching_results.tsv` is uploaded for the live leaderboard. Both TSVs are required in the final package. The packaged matching file must be byte-identical to the selected uploaded file; record hashes and the portal submission identifier.

## 2. Serialization contract

Matching header, with one literal tab:

```text
source1_entity_id\tmatched_entity_ids
```

Candidate header:

```text
source1_entity_id\tcandidate_entity_ids
```

The `\t` above denotes a real tab in the files, not two literal characters. Rows contain one S1 ID, a tab, and comma-separated target IDs with no spaces or list quoting. Empty output uses an empty second field. Use UTF-8 and a consistent newline convention.

Use a deterministic S1 order, preferably the input S1 manifest order, and stable sorted IDs within each list. Every S1 appears exactly once in each file, including France, unknown country labels, and records with empty candidates or no accepted matches.

## 3. Internal hard checks

Before packaging, verify all of the following on the compute PC:

- Exact filenames/headers, two fields per row, valid UTF-8, no accidental CSV delimiters, extra index column, BOM/header contamination, or literal missing-value strings.
- Output S1 set equals test S1 set, with exactly one row per entity.
- Every listed target exists in the correct test S2/S3 files and has a valid prefix; no training IDs used through a stale map.
- No duplicate target IDs in a row or duplicate S1 rows.
- Every final match is in the exported candidate set.
- Exported candidate pairs equal all unique pairs scored within this release's inference pipeline, including earlier matcher stages, ensemble members, and adaptive retrieval retries. Separate discarded experimental runs are not part of this release's candidate set.
- Model/feature/index versions belong to the same release; no train/test cache collision.
- Row and pair totals, empty-list counts, source/country coverage, multiplicity distributions, and candidate caps agree with the run report.
- Repeated serialization of the same ledger gives the same hashes; restarting/merging shards does not omit or duplicate S1 entities.

Perform candidate ledger comparison in sorted streaming form if necessary; do not require millions of Python sets in memory. Shape/distribution drift is a diagnostic requiring investigation, not permission to edit test matches manually.

## 4. Organizer validator

The following existing command is runnable once output files exist. Run from the repository root with the actual release directory substituted:

```powershell
python student_resource/utils/validate_submission.py --matching output/final/matching_results.tsv --candidate output/final/candidate_pairs.tsv --test-dir student_resource/dataset/test --check-ids
```

Require successful execution and inspect every warning. The supplied validator may use significant RAM when loading both files into string/set mappings. Run it after unloading training/index objects. If it cannot fit, retain the strict streaming checks, run the organizer's full format checks without ID loading, and separately validate matching IDs in a reduced-memory invocation. Record the exact checks performed; do not claim a full `--check-ids` run succeeded if it did not.

The validator can warn rather than fail for missing candidates and subset violations. Our release gates remain strict regardless of that exit status.

## 5. Reproduction requirements

The final code README must specify:

1. Tested OS, Python version, CPU/RAM/GPU/VRAM, dependency installation, and environment pins.
2. Where to put the original organizer data, plus expected schema and hash verification.
3. One clear end-to-end command to rebuild artifacts from training data and produce both outputs.
4. A faster inference-only command using the included final trained artifacts, when supplied.
5. Seeds, split generation, configuration, threshold selection, and exact checkpoint revisions.
6. Required disk/RAM and measured stage/total runtimes, with any hardware-dependent limits.
7. Resume behavior and validation/package commands.
8. Model and third-party licenses, parameter counts, and any nondeterminism relevant to output agreement.

Keep the code self-contained. Include trained tabular artifacts and any required preprocessing vocabularies/maps when feasible. If neural models are used, preserve tokenizer/config/weight access and exact revisions, and bundle permitted inference assets within organizer size constraints. Do not depend on a temporary machine path, an inaccessible private checkpoint, or an unspecified latest download. If package size rules prevent bundling necessary weights, obtain organizer guidance early and document a verifiable reconstruction route.

Reproduce from extracted packaged contents in a clean directory/environment on the compute PC. At minimum execute the complete pipeline on a synthetic/small approved fixture early, then perform the full release reproduction when time permits. The target is full regeneration with matching output hashes; if full regeneration is not completed, disclose exactly what was verified rather than calling the package fully reproduced.

## 6. Fill the provided methodology template

Keep [the supplied template](../student_resource/Documentation_template.md) unchanged as a reference. Create the filled final `Documentation_template.md` for packaging with **Team Name: RestoreBuildRun**. Add actual member names and submission date when available. Do not put planned metrics in the results section as if measured.

| Template section | Content to collect during implementation |
| --- | --- |
| Executive summary | Actual final pipeline and measured benefit, in 2–3 sentences |
| Problem analysis | Measured counts, missingness, singleton/multiplicity patterns, country shift, noise examples |
| Solution strategy | Pipeline stages and why the selected complexity fit the deadline/hardware |
| Candidate generation | Keys/channels, caps/adaptive policy, index design, actual matcher boundary |
| Blocking results | Total/mean/p95/p99 candidates, pair recall, macro ceiling, reduction denominator, runtime |
| Matching model | Feature families, architecture, parameter count, checkpoint/license, training samples/loss |
| Threshold selection | Exact macro metric, validation split, selected threshold, singleton handling |
| Results/error analysis | Development/holdout scores, slices, ablations, confidence intervals if run, failure modes |
| Conclusion | What worked, practical limits, no unmeasured claims about France or billion-scale deployment |
| Code artifacts | Entry points, package contents, input/output hashes, tested environment |
| Additional results | Small Pareto/ablation tables and provenance/license ledger |

Explicitly document the no-external-lookup policy, use of provided task data only, eligible pretrained model provenance if applicable, and any permitted deterministic augmentation. Explain leakage controls and separate training-only positive injection from measured retrieval recall.

## 7. Release checklist

- [ ] Team/member details correct; chosen release ID frozen.
- [ ] Both output files complete, immutable, and from the same run.
- [ ] Internal strict validation passed; candidate ledger equality verified.
- [ ] Organizer validator passed; all warnings reviewed and resolved or precisely documented.
- [ ] Exact metric and split method documented; results distinguish measured from planned.
- [ ] No unsupported external task data; model licenses and size verified.
- [ ] Runnable source/config/assets and tested dependency pins included.
- [ ] Extracted-package reproduction performed and scope recorded honestly.
- [ ] Final methodology template filled with actual results and limitations.
- [ ] Zip opens, contains required root paths, and has expected hashes.
- [ ] Leaderboard file uploaded early enough to resolve issues; portal status checked.
- [ ] Final archive submitted through the organizer's required channel; receipt saved.
- [ ] Selected uploaded and packaged matching files are byte-identical.

The user has requested planning, not a portal upload. This pass creates the plan only. During execution, the submitting team member must confirm the actual portal status and submission receipt; local file creation is not proof of submission.
