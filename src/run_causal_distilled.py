#!/usr/bin/env python3
"""Leakage-safe distillation of the Stocks V2 target construction.

The diagnostic formula reveals that the target rank is the exact-date
``date+2..date+5`` compounded ``f_2`` move.  This experiment uses that formula
only to create cleaner historical training labels.  Every predictor feature is
available at ``date <= t`` and the last five training dates are purged before
each future fold so the label horizon never overlaps validation.

The script is an experiment court only. It never writes or submits a Kaggle
prediction file.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd

from run_formula_candidate import exact_date_log_product
from run_pipeline import (
    KEY_COLUMNS,
    RAW_FEATURES,
    build_base_features,
    daily_rank_ic,
    sha256_file,
    within_date_rank,
)


BASELINE_OOF = 0.0894160876163553
HORIZON_END = 5
SEED = 20260712


@dataclass(frozen=True)
class DistilledFold:
    name: str
    nominal_train_end: int
    purged_label_end: int
    valid_start: int
    valid_end: int


FOLDS = (
    DistilledFold("purged_1195_to_future_1201_1450", 1200, 1195, 1201, 1450),
    DistilledFold("purged_1445_to_future_1451_1696", 1450, 1445, 1451, 1696),
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def validate_fold_contract(fold: DistilledFold) -> None:
    if fold.purged_label_end + HORIZON_END > fold.nominal_train_end:
        raise ValueError("purge does not contain the +5 target horizon")
    if fold.nominal_train_end >= fold.valid_start:
        raise ValueError("training/validation dates overlap")


def build_causal_features(train: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    empty_test = train[KEY_COLUMNS + RAW_FEATURES].iloc[:0].copy()
    base, _, names, _, _ = build_base_features(train, empty_test)
    # Replace the dataset-global min/max date scale with a fixed public-axis
    # scale, avoiding even target-free dependence on future fold extrema.
    date_index = names.index("date_scaled")
    base[:, date_index] = train["date"].to_numpy(dtype=np.float32) / 3000.0 - 0.5

    f2_log = np.log(pd.to_numeric(train["f_2"], errors="raise").to_numpy(dtype=np.float64))
    series = pd.Series(f2_log.astype(np.float32), copy=False)
    grouped = series.groupby(train["code"], sort=False, observed=True)
    extra: list[np.ndarray] = []
    extra_names: list[str] = []
    lags: dict[int, np.ndarray] = {}
    for lag in (1, 2, 3, 4, 5, 10, 20):
        values = grouped.shift(lag).fillna(0.0).to_numpy(dtype=np.float32)
        lags[lag] = values
        extra.append(values)
        extra_names.append(f"causal_log_f2_lag{lag}")
    extra.extend(
        [
            np.mean(np.column_stack([lags[k] for k in (1, 2, 3, 4, 5)]), axis=1).astype(np.float32),
            (lags[1] - lags[5]).astype(np.float32),
            (lags[5] - lags[20]).astype(np.float32),
        ]
    )
    extra_names.extend(["causal_log_f2_mean_lag1_5", "causal_log_f2_trend1_5", "causal_log_f2_trend5_20"])
    matrix = np.column_stack([base, *extra]).astype(np.float32, copy=False)
    if not np.isfinite(matrix).all():
        raise ValueError("causal feature matrix contains non-finite values")
    return matrix, names + extra_names


def run_fold(
    fold: DistilledFold,
    features: np.ndarray,
    distilled_rank: np.ndarray,
    formula_complete: np.ndarray,
    official_target: np.ndarray,
    dates: np.ndarray,
) -> dict[str, Any]:
    validate_fold_contract(fold)
    train_mask = (dates <= fold.purged_label_end) & formula_complete
    valid_mask = (dates >= fold.valid_start) & (dates <= fold.valid_end)
    valid_formula_mask = valid_mask & formula_complete
    if dates[train_mask].max() + HORIZON_END > fold.nominal_train_end:
        raise AssertionError("training label horizon crosses nominal boundary")
    if fold.nominal_train_end >= dates[valid_mask].min():
        raise AssertionError("strict future-date fold violated")

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
    train_set = lgb.Dataset(
        features[train_mask], label=distilled_rank[train_mask], free_raw_data=False
    )
    valid_set = lgb.Dataset(
        features[valid_formula_mask],
        label=distilled_rank[valid_formula_mask],
        reference=train_set,
        free_raw_data=False,
    )
    booster = lgb.train(
        params,
        train_set,
        num_boost_round=180,
        valid_sets=[valid_set],
        callbacks=[lgb.early_stopping(25, verbose=False), lgb.log_evaluation(0)],
    )
    prediction = booster.predict(
        features[valid_mask], num_iteration=booster.best_iteration
    ).astype(np.float32)
    valid_dates = dates[valid_mask]
    official_metric = daily_rank_ic(
        official_target[valid_mask], prediction, valid_dates
    )
    distilled_metric = daily_rank_ic(
        distilled_rank[valid_mask], prediction, valid_dates
    )
    receipt = {
        "fold": asdict(fold),
        "train_rows": int(train_mask.sum()),
        "valid_rows": int(valid_mask.sum()),
        "train_date_range": [int(dates[train_mask].min()), int(dates[train_mask].max())],
        "training_label_latest_source_date": int(dates[train_mask].max() + HORIZON_END),
        "validation_date_range": [int(valid_dates.min()), int(valid_dates.max())],
        "strict_future_date_order": True,
        "best_iteration": int(booster.best_iteration),
        "official_y_rank_ic": official_metric,
        "distilled_formula_rank_ic": distilled_metric,
    }
    del train_set, valid_set, booster, prediction
    gc.collect()
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, default=Path("data/raw/train_data.pkl"))
    parser.add_argument(
        "--receipt",
        type=Path,
        default=Path("artifacts/receipts/stocks_v2_causal_distilled.json"),
    )
    args = parser.parse_args()
    started = time.perf_counter()
    print("Loading train data for causal distillation...", flush=True)
    train = pd.read_pickle(args.train)
    dates = train["date"].to_numpy(dtype=np.int32)
    official_target = train["y"].to_numpy(dtype=np.float64)

    print("Constructing exact historical formula labels and <=t features...", flush=True)
    formula, complete = exact_date_log_product(train)
    distilled_rank = within_date_rank(np.where(complete, formula, 0.0), dates)
    features, feature_names = build_causal_features(train)

    folds = []
    for fold in FOLDS:
        print(f"Training {fold.name}...", flush=True)
        result = run_fold(
            fold, features, distilled_rank, complete, official_target, dates
        )
        folds.append(result)
        print(json.dumps(result["official_y_rank_ic"], sort_keys=True), flush=True)

    scores = [row["official_y_rank_ic"]["mean"] for row in folds]
    mean_score = float(np.mean(scores))
    minimum_score = float(min(scores))
    # "Material" is predeclared as +0.01 mean and no fold below the existing
    # 0.089416 OOF reference.
    gate = {
        "baseline_oof": BASELINE_OOF,
        "mean_fold_score": mean_score,
        "minimum_fold_score": minimum_score,
        "mean_margin": mean_score - BASELINE_OOF,
        "minimum_margin": minimum_score - BASELINE_OOF,
        "requires_mean_margin_at_least_0_01": bool(mean_score - BASELINE_OOF >= 0.01),
        "requires_every_fold_above_baseline": bool(minimum_score > BASELINE_OOF),
    }
    gate["passed"] = bool(
        gate["requires_mean_margin_at_least_0_01"]
        and gate["requires_every_fold_above_baseline"]
    )
    receipt = {
        "schema_version": "stocks-v2-causal-distilled-v1",
        "competition": "stocks-return-prediction-v-2",
        "created_at_utc": utc_now(),
        "external_submission_executed": False,
        "prediction_artifact_built": False,
        "information_contract": {
            "inference_features": "current-date and past feature rows only",
            "target_formula": "historical label only: exact date+2..date+5 f_2 log-product",
            "target_horizon_purge_dates": HORIZON_END,
            "validation": "strict expanding future dates",
            "test_future_rows_used": False,
        },
        "feature_count": len(feature_names),
        "features": feature_names,
        "folds": folds,
        "promotion_gate": gate,
        "decision": "PROMOTE_FOR_FINAL_FIT" if gate["passed"] else "KILL_NO_SUBMISSION",
        "input": {
            "path": str(args.train),
            "bytes": args.train.stat().st_size,
            "sha256": sha256_file(args.train),
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
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"receipt": str(args.receipt), "decision": receipt["decision"], "gate": gate}, indent=2), flush=True)


if __name__ == "__main__":
    main()
