# RestoreBuildRun — Amazon ML Challenge 2026

This repository contains the supplied business entity resolution challenge materials, the team's implementation plan, and the first executable baseline. **Implementation is authored; runtime verification, profiling, training, and inference are pending on the separate compute PC.** No measured model result is claimed yet.

## Run the implementation

Read the [implementation README](code/business_entity_resolution/README.md) and [current implementation status](docs/08_implementation_status.md). On the Windows compute PC, from the repository root:

```powershell
.\scripts\Run-Compute.ps1 -Dataset 'D:\AMLC\student_resource\dataset' -RunId baseline-001
```

Replace the dataset path with its actual location. The launcher installs dependencies, runs synthetic tests, trains/evaluates the baseline, benchmarks full-index query throughput, produces both TSVs, and validates them. All compute stays on that PC. No GPU is required for this first baseline.

## Recommended solution

Build a reproducible **blocking → pair matching → precision-aware set prediction** pipeline. Start with indexed lexical retrieval and a gradient-boosted matcher. Add multilingual embeddings to recover difficult candidates, then evaluate a neural pair scorer only if its improvement justifies its cost. Optimize both macro F₀.₅ and the number of candidates scored per Source 1 entity.

The critical requirements are: include every test Source 1 entity, support multiple matches and empty predictions, generalize to unseen France, avoid external business lookup, and export the actual candidate set scored by the matcher.

## Read the plan

| Document | Purpose |
| --- | --- |
| [01 — Requirements and decisions](docs/01_requirements_and_decisions.md) | Rules, deliverables, ambiguities, current evidence, and pending information |
| [02 — Data and validation](docs/02_data_and_validation.md) | Remote EDA, leakage prevention, exact metric, and validation design |
| [03 — Candidate generation](docs/03_candidate_generation.md) | Scalable blocking, recall/cost trade-offs, and candidate audit |
| [04 — Matching and decisions](docs/04_matching_and_decisions.md) | Features, model progression, negative sampling, thresholds, and France |
| [05 — Repository and compute](docs/05_repository_and_compute.md) | Two-PC workflow, planned code structure, environments, storage, and reproducibility |
| [06 — Experiments and milestones](docs/06_experiments_and_milestones.md) | Ordered implementation tasks, experiment matrix, stop conditions, and risks |
| [07 — Submission and documentation](docs/07_submission_and_documentation.md) | Release checks, exact package layout, and methodology write-up plan |

## Current evidence and boundaries

- Team: **RestoreBuildRun**.
- Configured remote: [Kar6677thik/AMLC_model](https://github.com/Kar6677thik/AMLC_model).
- At planning time, local `main` has no commits; challenge materials are untracked. Remote contents and repository visibility have not been checked.
- The seven provided TSV files occupy approximately **2.52 GB in decimal units**, based on filesystem metadata. Actual row counts, singleton rates, label integrity, and noise distributions remain to be measured on the compute PC.
- Compute PC: **Windows ThinkPad P16, Intel Ultra 7, 32 GB RAM**, substantial free storage; GPU model and VRAM remain unverified.
- User deadline: **tonight, 2026-09-27 at 23:59**. The schedule assumes **Asia/Kolkata (IST)** from the workspace timezone; verify the portal timezone. Planning clock check was approximately 08:57 IST, leaving about 15 hours.
- The deadline plan prioritizes a runnable lexical/tabular baseline and early full inference. Neural work is optional and gated by measured completion time. See [the schedule](docs/06_experiments_and_milestones.md).
- Documents 01–07 preserve the design plan. Use the implementation README for commands that now exist; the actual baseline intentionally omits the deferred neural stages and some longer-term validation experiments.
- No data, code, documents, or outputs have been pushed to the remote in these passes. Ignore rules now exclude challenge data and generated artifacts.

The original organizer materials remain unchanged. The next gate is the synthetic test suite on the compute PC, followed by the first measured baseline run.
