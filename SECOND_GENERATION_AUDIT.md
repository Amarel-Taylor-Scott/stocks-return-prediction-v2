# Stocks V2 second-generation audit

## Decision

Two distinct results were separated rather than blended into one claim:

1. **Target-construction diagnostic — do not submit.** The rank of `y(code,date)` is virtually identical to the
   exact-date product of `f_2(code,date+2)` through `f_2(code,date+5)`. This explains the approximately `0.996`
   leaders, but consumes later test rows and is noncausal under a live-forecast interpretation. The parent decision
   is to retain it only as research evidence.
2. **Causal distillation — local candidate ready.** The same construction is used only to create cleaner historical
   labels. Inference uses current and past feature rows, and five dates are purged at every training boundary.

No external submission was executed for either route.

## Exact target diagnosis

The correct join is `(code, exact date+k)`, not the next observed row for a code:

```text
q(code,date) = sum(log(f_2(code,date+k))) for k in {2,3,4,5}
```

`q` is rank-equivalent to the four-session product. On complete rows it scored:

| Strict validation dates | Exact-date Rank IC | Next-observation Rank IC | Coverage |
|---|---:|---:|---:|
| 1201-1450 | 0.999997 | 0.992367 | 99.37% |
| 1451-1696 | 0.999996 | 0.997689 | 99.74% |

Neutral scoring for sparse date holes gives all-row fold scores `0.995822` and `0.997409`.

Matched falsifiers behaved as required:

| Placebo | Fold 1 | Fold 2 |
|---|---:|---:|
| Exact causal mirror, dates t-5...t-2 | -0.00674 | -0.02649 |
| Rotate formula scores across codes within each date | -0.00074 | -0.00078 |
| Wrong future window t+1...t+4 | 0.68921 | 0.68174 |
| Wrong future window t+3...t+6 | 0.68890 | 0.68144 |

The code-rotation control preserves every date's prediction distribution while breaking code identity, so the
near-perfect result cannot be attributed to a date-level scale artifact.

## Compliance boundary

The official description says to predict “with the features data in the test set,” and no sequential serving rule
is stated. The project is nevertheless described as future-return prediction, and its only specific rule is
“Don't cheat!” Later test dates would not exist at live inference time. Because this ambiguity cannot be resolved
from the published text alone, the future-row ZIP was deleted after its receipt was recorded and is not an eligible
handoff candidate.

## Causal distilled court

The distilled member uses the formula only as a historical label. Its 40 predictors contain same-date raw/rank
features and past per-code lags/trends. For fold 1, labels stop at date 1195 and their latest source feature is date
1200; validation starts at 1201. For fold 2, labels stop at 1445/source 1450; validation starts at 1451.

| Model | Fold 1201-1450 | Fold 1451-1696 | Two-fold mean |
|---|---:|---:|---:|
| Existing v1 OOF reference | — | — | 0.089416 |
| Causal distilled | **0.108384** | **0.097766** | **0.103075** |

The predeclared gate required every fold above `0.089416` and at least `+0.01` mean improvement. It passed with
`+0.013659` mean and `+0.008350` worst-fold margins.

## Final local artifact

- ZIP: `artifacts/submissions/stocks_v2_causal_distilled_submission.zip`
- ZIP SHA-256: `7631c79b81f7926d803aa80dc4fc8da2a6ea27f562701e48e397dbe57069e215`
- Rows: `4,879,631`
- Final rounds: `116`, the median of fold optima `152` and `80`
- Training label dates: `0-1696`; latest source feature date: `1701`; first test date: `1702`
- Audit: exact columns, IDs, keys, row count, uniqueness, numeric/finite/nonconstant predictions, one ZIP member,
  and passing ZIP CRC
- External submission: **not executed**

Prepared command, intentionally not run:

```bash
kaggle competitions submit -c stocks-return-prediction-v-2 -f artifacts/submissions/stocks_v2_causal_distilled_submission.zip -m "causal distilled +2..+5 target; purged future-date CV 0.103075"
```

Receipts:

- `artifacts/receipts/stocks_v2_formula_candidate.json`
- `artifacts/receipts/stocks_v2_causal_distilled.json`
- `artifacts/receipts/stocks_v2_causal_distilled_final.json`

Reproduction tests: `python3 -m pytest -q tests/test_pipeline.py tests/test_formula_candidate.py` (`9 passed`).

## Public transfer diagnosis

Submission `54616087` completed at `0.08961`, below v1's `0.09057`. The distilled
OOF-to-public gap was `-0.013465`, while v1 transferred almost exactly. Artifact
integrity was not the issue: schema, keys, ranks, hash, and ZIP CRC all remained
valid. The evidence points to historical-world selection/shift: distilled labels
made the purged historical folds easier but did not improve the hidden test regime.

The two test predictions are related but not redundant: mean daily rank
correlation `0.84839`, sign disagreement `16.14%`, top-10% overlap `46.78%`, and
top-1% overlap `15.24%`.

One aligned OOF court therefore selected blend weight using fold 1 only and
opened fold 2 afterward. The selected `90%` distilled / `10%` v1 blend improved:

| | Fold 1 | Sealed fold 2 |
|---|---:|---:|
| v1 | 0.082652 | 0.097344 |
| guarded blend | **0.108731** | **0.098300** |

The predeclared mean and worst-fold gates passed by `+0.013518` and `+0.015648`.
The resulting local ZIP is
`artifacts/submissions/stocks_v2_v1_distilled_blend_submission.zip`, SHA-256
`5899f77306d59f969b754a70271e1a1c54b9ac3bf2459c167100d84c9d0bee13`.
It has not been submitted. Recommendation: use at most one remaining slot for
this probe and stop the lane if it does not beat v1.
