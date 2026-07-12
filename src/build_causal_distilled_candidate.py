#!/usr/bin/env python3
"""Final-fit the promoted causal distilled Stocks V2 candidate locally.

Requires the two-fold promotion receipt from ``run_causal_distilled.py``. The
script writes an audited ZIP but never calls Kaggle.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from run_formula_candidate import exact_date_log_product
from run_pipeline import (
    KEY_COLUMNS,
    PREDICTION,
    RAW_FEATURES,
    audit_submission,
    atomic_json,
    build_base_features,
    sha256_file,
    within_date_rank,
    write_submission_zip,
)


SEED = 20260712
FINAL_LABEL_END = 1696
FINAL_LABEL_SOURCE_END = 1701


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def build_combined_causal_features(
    train: pd.DataFrame, test: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    base_train, base_test, names, _, _ = build_base_features(train, test)
    n_train = len(train)
    all_rows = pd.concat(
        [train[KEY_COLUMNS + RAW_FEATURES], test[KEY_COLUMNS + RAW_FEATURES]],
        ignore_index=True,
    )
    date_index = names.index("date_scaled")
    fixed_date = all_rows["date"].to_numpy(dtype=np.float32) / 3000.0 - 0.5
    base_train[:, date_index] = fixed_date[:n_train]
    base_test[:, date_index] = fixed_date[n_train:]

    f2_log = np.log(pd.to_numeric(all_rows["f_2"], errors="raise").to_numpy(dtype=np.float64))
    series = pd.Series(f2_log.astype(np.float32), copy=False)
    grouped = series.groupby(all_rows["code"], sort=False, observed=True)
    lag_values: dict[int, np.ndarray] = {}
    extra: list[np.ndarray] = []
    extra_names: list[str] = []
    for lag in (1, 2, 3, 4, 5, 10, 20):
        values = grouped.shift(lag).fillna(0.0).to_numpy(dtype=np.float32)
        lag_values[lag] = values
        extra.append(values)
        extra_names.append(f"causal_log_f2_lag{lag}")
    extra.extend(
        [
            np.mean(np.column_stack([lag_values[k] for k in (1, 2, 3, 4, 5)]), axis=1).astype(np.float32),
            (lag_values[1] - lag_values[5]).astype(np.float32),
            (lag_values[5] - lag_values[20]).astype(np.float32),
        ]
    )
    extra_names.extend(["causal_log_f2_mean_lag1_5", "causal_log_f2_trend1_5", "causal_log_f2_trend5_20"])
    extra_matrix = np.column_stack(extra).astype(np.float32, copy=False)
    x_train = np.column_stack([base_train, extra_matrix[:n_train]]).astype(np.float32, copy=False)
    x_test = np.column_stack([base_test, extra_matrix[n_train:]]).astype(np.float32, copy=False)
    if not np.isfinite(x_train).all() or not np.isfinite(x_test).all():
        raise ValueError("combined causal features contain non-finite values")
    del all_rows, base_train, base_test, extra, extra_matrix, lag_values
    gc.collect()
    return x_train, x_test, names + extra_names


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--artifacts-dir", type=Path, default=Path("artifacts"))
    parser.add_argument(
        "--promotion-receipt",
        type=Path,
        default=Path("artifacts/receipts/stocks_v2_causal_distilled.json"),
    )
    args = parser.parse_args()
    started = time.perf_counter()
    promotion = json.loads(args.promotion_receipt.read_text())
    if promotion.get("decision") != "PROMOTE_FOR_FINAL_FIT" or not promotion["promotion_gate"]["passed"]:
        raise RuntimeError("causal candidate lacks a passing promotion receipt")
    iterations = max(50, int(round(np.median([row["best_iteration"] for row in promotion["folds"]]))))

    train_path = args.data_dir / "train_data.pkl"
    test_path = args.data_dir / "test_data.pkl"
    sample_path = args.data_dir / "sample_submission.csv"
    print("Loading train/test for promoted causal final fit...", flush=True)
    train = pd.read_pickle(train_path)
    test = pd.read_pickle(test_path)
    dates = train["date"].to_numpy(dtype=np.int32)
    formula, complete = exact_date_log_product(train)
    label = within_date_rank(np.where(complete, formula, 0.0), dates)
    train_mask = (dates <= FINAL_LABEL_END) & complete
    if int(dates[train_mask].max()) + 5 != FINAL_LABEL_SOURCE_END:
        raise AssertionError("final label horizon/purge contract changed")

    print("Building current/past-only features across the train/test boundary...", flush=True)
    x_train, x_test, feature_names = build_combined_causal_features(train, test)
    params = {
        "objective": "regression",
        "metric": "l2",
        "learning_rate": 0.06,
        "num_leaves": 31,
        "min_data_in_leaf": 1500,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.85,
        "bagging_freq": 1,
        "lambda_l1": 0.2,
        "lambda_l2": 3.0,
        "max_bin": 127,
        "seed": SEED,
        "feature_fraction_seed": SEED,
        "bagging_seed": SEED,
        "num_threads": min(16, os.cpu_count() or 4),
        "verbosity": -1,
        "force_col_wise": True,
    }
    print(f"Training final causal distilled model for {iterations} rounds...", flush=True)
    dataset = lgb.Dataset(x_train[train_mask], label=label[train_mask])
    booster = lgb.train(params, dataset, num_boost_round=iterations)
    raw_prediction = booster.predict(x_test, num_iteration=iterations).astype(np.float32)
    test_dates = test["date"].to_numpy(dtype=np.int32)
    prediction = within_date_rank(raw_prediction, test_dates)

    sample = pd.read_csv(sample_path)
    prediction_frame = test[KEY_COLUMNS].copy()
    prediction_frame[PREDICTION] = prediction
    submission = sample.drop(columns=[PREDICTION]).merge(
        prediction_frame, on=KEY_COLUMNS, how="left", validate="one_to_one"
    )
    submission = submission[list(sample.columns)]
    audit = audit_submission(submission, sample, test)
    artifact = write_submission_zip(
        submission,
        args.artifacts_dir / "submissions" / "stocks_v2_causal_distilled_submission.csv",
        args.artifacts_dir / "submissions" / "stocks_v2_causal_distilled_submission.zip",
    )
    receipt = {
        "schema_version": "stocks-v2-causal-distilled-final-v1",
        "competition": "stocks-return-prediction-v-2",
        "created_at_utc": utc_now(),
        "external_submission_executed": False,
        "decision": "LOCAL_CANDIDATE_READY_NO_EXTERNAL_SUBMIT",
        "promotion_receipt": {
            "path": str(args.promotion_receipt),
            "sha256": sha256_file(args.promotion_receipt),
            "gate": promotion["promotion_gate"],
        },
        "information_contract": {
            "inference_features": "date<=t only, including train-tail lags for test",
            "training_labels": "historical exact date+2..date+5 f_2 formula ranks",
            "final_training_label_dates": [int(dates[train_mask].min()), int(dates[train_mask].max())],
            "latest_feature_date_used_by_training_labels": FINAL_LABEL_SOURCE_END,
            "earliest_test_date": int(test.date.min()),
            "test_future_rows_used": False,
        },
        "feature_count": len(feature_names),
        "features": feature_names,
        "iterations": iterations,
        "train_rows": int(train_mask.sum()),
        "submission_audit": audit,
        "submission_artifact": artifact,
        "handoff": {
            "recommended_message": "causal distilled +2..+5 target; purged future-date CV 0.103075",
            "submit_command": (
                "kaggle competitions submit -c stocks-return-prediction-v-2 "
                f"-f {artifact['zip_path']} "
                "-m \"causal distilled +2..+5 target; purged future-date CV 0.103075\""
            ),
        },
        "inputs": {
            "train_sha256": sha256_file(train_path),
            "test_sha256": sha256_file(test_path),
            "sample_sha256": sha256_file(sample_path),
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "lightgbm": lgb.__version__,
            "seed": SEED,
        },
        "runtime_seconds": float(time.perf_counter() - started),
    }
    receipt_path = args.artifacts_dir / "receipts" / "stocks_v2_causal_distilled_final.json"
    atomic_json(receipt_path, receipt)
    print(json.dumps({"receipt": str(receipt_path), "artifact": artifact["zip_path"], "sha256": artifact["zip_sha256"], "external_submission_executed": False}, indent=2), flush=True)


if __name__ == "__main__":
    main()
