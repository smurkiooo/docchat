# Evaluation report

Run: `2d08e5a19b6449a192370f2eb5dcee0d` · split: dev · backend: real

Synthetic, jointly authored fixtures; not an independent generalization benchmark.
No statistical significance or production-quality claim is made.

All methods use identical queries, chunks, evidence labels and output cutoffs.
Hybrid can use the union of both candidate pools; its compute budget is larger.

| Mode | n | Hit@1 | Evidence recall@6 | Complete@6 | MRR@6 | Document macro Complete@6 |
|---|---:|---:|---:|---:|---:|---:|
| bm25 | 16 | 0.875 | 0.938 | 0.938 | 0.896 | 0.938 |
| dense | 16 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| hybrid | 16 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |

Answerable cases: 16; unanswerable: 8.
Oracle complete evidence after parsing/chunking: 16/16.

## Gate (deployed context only)

Thresholds used: cosine >= 0.8 or lexical coverage >= 0.5.
False rejections: 1/16 answerable.
False acceptances: 4/8 unanswerable.
Accepted with incomplete evidence: 0.

Median shared ranking pass: 31.1 ms. One warm-up per document; includes all enabled rankings; not per-mode latency.

Per-case rankings, all @1/3/6 metrics, document/tag breakdowns, config, hashes, chunks and raw predictions are in the adjacent JSON artifact.

## Generation

Generated 28/28 records (single-turn and dialogues).
Citation presence and refusal templates are diagnostics, not correctness scores.
Automated rubric review is complete. It is not an independent human review; see the adjacent review JSONL and review-summary JSON.

Model revision: d128750597153bb5987e10b1c3493a34e5a4502a.
See docs/EVALUATION.md for the protocol and limitations.
