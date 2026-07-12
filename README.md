# Stocks Return Prediction V2

Competition-specific, leakage-audited pipeline for Kaggle's `stocks-return-prediction-v-2` challenge.

## Evidence boundary

- Only the V2 competition files are used. V1 and V3 assets are not inputs or evidence.
- Validation is expanding-window by date. Every validation date is later than every training date.
- Cross-sectional ranks use only features from the same date, which are available at inference.
- Lag features use current or earlier feature rows only.
- The stock-code target prior is expanding for training rows and frozen at each validation boundary. It never uses a validation or future target.
- The official sample submission contains nonconstant, target-shaped predictions. It is audited and retained as a separate **host reference**, never presented as model or CV evidence.

## Pipeline

The system trains three deliberately different members:

1. A past-only stock-code prior baseline.
2. Ridge regression over raw, cross-sectional-rank, and causal lag features.
3. LightGBM over the same features.

OOF predictions from two expanding temporal folds select a nonnegative rank blend. The final output is projected to within-date ranks because the official metric is mean daily Spearman correlation (Rank IC).

## Run

```bash
python src/run_pipeline.py \
  --data-dir data/raw \
  --artifacts-dir artifacts
```

Fast synthetic verification:

```bash
pytest -q
```

The run writes an exact JSON receipt, an audited model submission ZIP, and a
small submission-command handoff. The training script itself never submits to
Kaggle; platform submissions are separate, explicit transactions.

## Verified run — 2026-07-12

| Member | Dates 1201–1450 | Dates 1451–1701 |
|---|---:|---:|
| Past-only code prior | 0.00110 | 0.01913 |
| Ridge | 0.07349 | 0.07932 |
| LightGBM | 0.08034 | 0.09785 |

The selected 80% LightGBM / 20% Ridge blend reached **0.089416 mean daily Rank IC** over 501 strictly future validation dates and 1,699,763 rows. The trained candidate contains all 4,879,631 required rows and passed exact columns, IDs, keys, uniqueness, numeric, finite, nonconstant, ZIP CRC, and line-count checks.

- Candidate: `artifacts/submissions/stocks_v2_model_blend_submission.zip`
- Candidate SHA-256: `da532638f8de48ae24a5d5d45a386621d1f617c614f17c6f63772cdf8184a56a`
- Exact receipt: `artifacts/receipts/stocks_v2_run.json`
- Live contract: `artifacts/receipts/live_competition_contract.json`
- Official host-reference diagnostic: **−0.00028 public** (submission
  `54614975`), falsifying the apparent sample-file shortcut.
- Leakage-safe trained blend: **0.09057 public** (submission `54615059`), close
  to its 0.089416 future-date OOF estimate.
- Platform receipt: `artifacts/receipts/kaggle_submissions_2026-07-12.json`

## Live competition contract

- Deadline: 2026-07-15 16:00 UTC
- Input: `code`, `date`, `f_0` through `f_6`; train additionally has `y`
- Submission: `id`, `code`, `date`, `y_pred`
- Metric: average across dates of Spearman correlation between predicted and true within-date ranks
- Reward: Kudos
- Account status at intake: entered, with no previous submissions

Official URL: <https://www.kaggle.com/competitions/stocks-return-prediction-v-2>

## Second generation — causal target distillation

Train-only forensics identified the exact rank target construction as the `f_2`
product at dates `t+2...t+5`. A static future-row version scores almost `1.0`,
but is quarantined as diagnostic-only and must not be submitted because it is
noncausal under a live-forecast interpretation.

The leakage-safe distillation uses that construction only for historical
labels, purges the full five-date horizon, and predicts from same-date/past
features. Its strict folds scored `0.108384` and `0.097766` (mean `0.103075`),
versus v1 OOF `0.089416`. The final local ZIP passed every schema and archive
check; no submission was executed. See `SECOND_GENERATION_AUDIT.md` and
`artifacts/receipts/stocks_v2_causal_distilled_final.json`.

The causal distilled submission later scored `0.08961` public versus v1
`0.09057`. An aligned, fold-1-selected / fold-2-sealed court promoted a guarded
`90%` distilled + `10%` v1 blend locally; it remains unsubmitted. See the audit
for the transfer diagnosis and stop rule.
