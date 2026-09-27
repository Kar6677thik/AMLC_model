# RestoreBuildRun — Amazon ML Challenge 2026

This repository contains the challenge materials, plans and matching pipeline. **Baseline-001 scored 0.806 on the submitted run (user reported). Optimized-002 improved development F0.5 from 0.84050 to 0.86730.** A subsequent probe showed that string preparation/ranking dominates its runtime. The current experiment preserves full v2 retrieval, removes unused ranking work, and reuses saved fit features to compare GPU models and an ensemble. New results remain to be measured.

## Run the implementation

Read the [quality-first plan](docs/11_quality_first_plan.md) and [implementation README](code/business_entity_resolution/README.md). On the Windows compute PC, from the repository root:

```powershell
.\scripts\Run-Quality.ps1 -Dataset 'D:\AMLC\student_resource\dataset' -StudyId quality-004 -Phase Evaluate
```

Replace the dataset path. This runs tests, verifies CUDA, reuses optimized-002's saved fit arrays, compares full-quality matchers and benchmarks the best model. It stops before prediction. After reviewing `runs/quality-004/quality.json`, use `-Phase Predict -SkipInstall` if a model is selected. Preserve optimized-002's arrays and all earlier outputs. All compute stays on that PC.

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
| [09 — Baseline analysis and optimization](docs/09_baseline_analysis_and_optimization.md) | Measured bottlenecks, implemented improvements, GPU settings and next-run commands |
| [10 — Faster retrieval](docs/10_fast_retrieval_experiment.md) | Optimized-002 results, conditional retrieval, probe selection and deadline gates |
| [11 — Quality-first plan](docs/11_quality_first_plan.md) | Probe findings, compact ranking, saved-feature refits, ensemble and deadline-aware selection |

## Current evidence and boundaries

- Team: **RestoreBuildRun**.
- Configured remote: [Kar6677thik/AMLC_model](https://github.com/Kar6677thik/AMLC_model).
- Baseline reports record Git commit `5d4b1e42a1cf6db1e1754f4033bf41970f16a406`. Remote visibility has not been checked.
- Baseline reports contain **1,732,544 test anchors** and measured full-index throughput of **107.43 anchors/second**. The user reported about five hours for the complete run.
- Compute PC: **Windows ThinkPad P16, Intel Ultra 7, 32 GB RAM**, NVIDIA RTX PRO 1000 Blackwell laptop GPU with **8,151 MiB dedicated VRAM**, driver 596.58; about 258 GB free disk space reported.
- User deadline: **tonight, 2026-09-27 at 23:59**. The schedule assumes **Asia/Kolkata (IST)** from the workspace timezone; verify the portal timezone. Planning clock check was approximately 08:57 IST, leaving about 15 hours.
- The deadline plan prioritizes a runnable lexical/tabular baseline and early full inference. Neural work is optional and gated by measured completion time. See [the schedule](docs/06_experiments_and_milestones.md).
- Documents 01–07 preserve the design plan. Use the implementation README for commands that now exist; the actual baseline intentionally omits the deferred neural stages and some longer-term validation experiments.
- The optimization changes have not been pushed by this implementation pass. Ignore rules exclude challenge data and generated artifacts.

The original organizer materials remain unchanged. The next gate is runtime verification and a measured optimized run on the compute PC.
