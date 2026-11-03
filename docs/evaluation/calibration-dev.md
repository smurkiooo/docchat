# Evaluation report

Run: `2a67ed024cf74a74b1de3e35258e5135` · split: dev · backend: real

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

Thresholds used: cosine >= 0.72 or lexical coverage >= 0.5.
False rejections: 0/16 answerable.
False acceptances: 7/8 unanswerable.
Accepted with incomplete evidence: 0.

## Dev-only calibration (not applied to the gate counts above)

Selected cosine 0.8, coverage 0.5; balanced error 0.2812; false rejections 1, false acceptances 4.
Full grid and selection rule are in JSON. Always-pass/always-reject baselines each have balanced error 0.5 when both classes are present.

Median shared ranking pass: 31.1 ms. One warm-up per document; includes all enabled rankings; not per-mode latency.

Per-case rankings, all @1/3/6 metrics, document/tag breakdowns, config, hashes, chunks and raw predictions are in the adjacent JSON artifact.

## Generation

Not run. No answer-quality or citation-correctness score is claimed.

Model revision: d128750597153bb5987e10b1c3493a34e5a4502a.
See docs/EVALUATION.md for the protocol and limitations.
