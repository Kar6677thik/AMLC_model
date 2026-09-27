# 03 — Scalable candidate generation

## 1. Objective and pipeline boundary

Retrieve a small set of plausible S2/S3 records for each S1, preserving as much achievable macro F₀.₅ as possible. The organizer explicitly considers candidate efficiency in final ranking.

```text
Raw records
    ↓ loss-aware normalization
Source/country indexes
    ↓ independent bounded retrieval channels
Union + deduplication + retrieval-only budget policy
    ↓ persist immutable candidate ledger
Pair features + matching model(s)
    ↓ set decision
matching_results.tsv

candidate_pairs.tsv = exact unique pairs entering the matching stage
```

A pair scored by the gradient-boosted matcher remains a candidate even if a later neural scorer sees only an uncertain subset. Do not trim the reported candidates to accepted matches or to that neural subset. Treat any supervised identity scorer as part of matching for a conservative, auditable boundary. Retrieval similarities used for indexed ranking are logged separately.

## 2. Normalization views

Keep the raw fields and generate several views rather than destructively replacing text:

- Unicode normalization, case folding, whitespace/punctuation normalization.
- Accent-preserving and accent-folded versions; never remove non-Latin text globally.
- Original-order name tokens, sorted token view, initials/acronym view, character n-grams.
- Conservative legal-suffix/abbreviation variants learned or defined from supplied guidance and training data. Preserve the original name alongside them.
- Address token/n-gram view; numeric token sets; possible house/unit/postal components with parser confidence.
- Country normalized as an open string label with an unknown/missing path.

Do not remove every number, expand ambiguous abbreviations globally, or assume that `St` always means street. A legal suffix removal can collapse distinct entities; a house-number difference can distinguish branches. Retain uncertainty and original evidence.

## 3. Deadline baseline retrieval

Build **disk-backed or compact integer posting indexes**, partitioned by source and, if the audit supports it, country. Country-only partitioning is not a complete blocker: each country can still be huge. Use composite keys and bounded postings inside it.

| Channel | Key/ranking | Intended coverage | Controls |
| --- | --- | --- | --- |
| Exact normalized composite | Name + address, with alternate text views | Easy identity-preserving variations | Detect collisions; never accept solely from channel membership |
| Rare name tokens | Rare informative name tokens, refined by address tokens | Reordered words and partial names | Exclude stop/common tokens; bounded postings/intersections |
| Name + numeric/address evidence | Name prefix/token with possible postal/house/locality evidence | Same name in different locations | Use as one channel, never the only route |
| Character n-gram retrieval | Name and address retrieval scores kept separate | Typos, punctuation, morphology | Bounded top-k sparse search, no dense all-pairs matrix |
| Initials/abbreviation view | Acronym plus supporting address token | Shortened names | Aggressive collision controls |

Start with the first three channels for the fastest runnable implementation, then add character retrieval if its measured incremental recall warrants the index cost. Avoid implementing every channel before the first end-to-end run.

Compute lexical scores using overlap/rarity weights or TF-IDF in bounded blocks. A whole-country sparse matrix product can still produce an enormous intermediate result; use inverted retrieval with bounded postings or a verified top-k implementation. No exhaustive nested loops across S1 and all targets. Record total posting visits and candidate-stage wall time as well as emitted candidates.

For very common names or huge exact buckets, intersect independent address/rare-token keys before ranking. Do not silently truncate a posting list by ID order. Log every truncation, saturation, fallback, and resulting candidate cap.

## 4. Country and source coverage

Audit positive pairs before treating known country mismatch as impossible. Start with matching-country indexes where supported, but retain explicit handling for missing/unknown countries. If training evidence shows mismatches, add a small global rare-key channel and assess its incremental recall and false candidates.

Enumerate country labels from each input. France receives the same generic indexing and fallback logic without special labeled thresholds. Unknown future labels must also work.

Retrieve separately from S2 and S3, then merge. This prevents a large S2 result list from starving S3. Permit multiple matches from the same source. Duplicate texts may share an encoding/cache entry, but all distinct original target IDs must be recoverable and count toward the exported candidate total.

## 5. Candidate budgets and adaptive search

Initial **development grid**, not a production promise: total caps of 10, 20, 40, and 80 candidates per S1 after channel union. Use a broader budget on a bounded diagnostic sample to estimate what capped retrieval misses. Inspect true multiplicity before any cap: a cap below the number of genuine matches makes complete recovery impossible.

Combine channel ranks with a deterministic rule such as reciprocal-rank fusion and retain channel provenance. Set minimum S2/S3 coverage where supported; remaining slots follow evidence. Do not force candidates for every S1 if no channel has credible evidence.

Adaptive expansion can use retrieval evidence: ambiguous/common names, disagreement across channels, no strong composite hit, or a saturated candidate list. It must have an explicit maximum search budget and be tuned on development data. If matching scores trigger a retry, union both the first and expanded scored sets into the exported candidates and replay that behavior during validation.

Provisional aspiration: micro candidate recall around 99% and macro oracle ceiling close to 1.0, with the smallest feasible count. These are targets to investigate, not acceptance requirements that justify missing tonight's deadline. A measured lower-recall baseline remains the fallback.

Use a Pareto table containing candidate recall, full recovery, macro oracle ceiling, end-to-end F₀.₅, mean/p95/p99 candidates, total scored pairs, stage runtime, and memory. Do not reduce candidates merely because the overall reduction ratio looks impressive; tiny fractions of a quadratic universe can still be expensive.

## 6. Optional multilingual dense channel

Only start after a runnable baseline and full-size throughput estimate. Encode each distinct record text once, cache embeddings, and query an approximate nearest-neighbor index. Start with one compact combined representation; separate name/address encoders multiply storage and work.

The compact candidate is `intfloat/multilingual-e5-small`: its model card declares MIT and 384-dimensional embeddings. Follow its documented prefixes and pooling; for symmetric entity similarity, start with `query:` on both records and validate the choice. Cosine values are retrieval evidence, not calibrated identity probabilities. [Official model card](https://huggingface.co/intfloat/multilingual-e5-small/raw/main/README.md)

Use source/country ANN partitions that fit memory; benchmark the actually available Windows library. HNSW is a candidate if it installs cleanly and fits RAM. IVF-based compressed indexes are a later option if needed; avoid an index-engine migration on deadline day. Record ANN search parameters, training sample where applicable, and seed. [FAISS index-selection guidance](https://github.com/facebookresearch/faiss/wiki/Guidelines-to-choose-an-index)

Validate ANN recall against exact search on a small held-out query sample. That is an index-quality diagnostic, distinct from recall of true business matches. Exact flat search over all targets for every query is not the scalable production solution.

Dense retrieval must add useful matches missed by lexical channels. Related businesses often have similar embeddings, so dense-only candidates still require name/address identity evidence. If full encoding and index construction cannot finish within the protected runtime budget, keep the lexical release.

## 7. Scaling and memory discipline

Let `N = N2 + N3` and `M = sum_i |C_i|`.

- Index build cost should be tied to records/tokens, not all pairs.
- Matching cost is proportional to `M`; retrieval's internal posting/ANN work must also be measured.
- Float16 embedding storage is approximately `2*N*d` bytes, excluding the ANN structure, query vectors, and runtime copies. One million 384-dimensional vectors use roughly 0.768 GB decimal before overhead; some indexes require float32 copies.
- Store pair endpoints as integer indices plus compact score/provenance fields. Keep text in one record table, not duplicated in every feature row.
- Partition large stages, write shards, and release indexes/arrays before the next stage when RAM is tight.
- No billion-record scalability claim based on the supplied files alone. Explain bounded retrieval and shardability, and report measured throughput on this dataset.

## 8. Candidate audit artifacts

For each run persist: input/index hashes, normalization version, source maps, retrieval configuration, per-pair channel/rank scores, final candidate ledger, scoring-ledger hash/count, and final output hashes.

Assert `unique(scored_pair_keys) == exported_candidate_pair_keys` and `matched_pair_keys ⊆ scored_pair_keys`. Ensembles share the same immutable candidate set; if different models use different sets, export their union and account for it in candidate statistics.

For every retrieval miss sampled from validation, identify the channel that should have recovered it and the failing step: parser, partition, posting cutoff, top-k cap, ANN miss, or representation. Fix the dominant measurable cause before adding another model.
