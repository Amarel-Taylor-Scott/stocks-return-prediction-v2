#!/usr/bin/env python3
"""Leakage-safe V2 stock-return training, blending, and submission audit.

The script intentionally does not call Kaggle's submission endpoint.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import platform
import sys
import time
import zipfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import lightgbm as lgb
import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.linear_model import Ridge


COMPETITION = "stocks-return-prediction-v-2"
KEY_COLUMNS = ["code", "date"]
RAW_FEATURES = [f"f_{i}" for i in range(7)]
TARGET = "y"
PREDICTION = "y_pred"
SEED = 20260712


@dataclass(frozen=True)
class FoldSpec:
    name: str
    train_end: int
    valid_start: int
    valid_end: int


FOLDS = (
    FoldSpec("expanding_1200_to_1450", 1200, 1201, 1450),
    FoldSpec("expanding_1450_to_1701", 1450, 1451, 1701),
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def assert_input_contract(train: pd.DataFrame, test: pd.DataFrame, sample: pd.DataFrame) -> None:
    expected_train = KEY_COLUMNS + RAW_FEATURES + [TARGET]
    expected_test = KEY_COLUMNS + RAW_FEATURES
    expected_sample = ["id", "code", "date", PREDICTION]
    if list(train.columns) != expected_train:
        raise ValueError(f"unexpected train columns: {list(train.columns)}")
    if list(test.columns) != expected_test:
        raise ValueError(f"unexpected test columns: {list(test.columns)}")
    if list(sample.columns) != expected_sample:
        raise ValueError(f"unexpected sample columns: {list(sample.columns)}")
    if train.duplicated(KEY_COLUMNS).any() or test.duplicated(KEY_COLUMNS).any():
        raise ValueError("duplicate (code, date) keys")
    if sample.duplicated(KEY_COLUMNS).any() or not sample["id"].is_unique:
        raise ValueError("sample keys or IDs are not unique")
    probe = sample[KEY_COLUMNS].merge(test[KEY_COLUMNS], on=KEY_COLUMNS, how="outer", indicator=True)
    if len(probe) != len(test) or not (probe["_merge"] == "both").all():
        raise ValueError("sample and test key sets differ")
    if train.isna().any().any() or test.isna().any().any():
        raise ValueError("input contains missing values")
    if int(train["date"].max()) >= int(test["date"].min()):
        raise ValueError("train/test dates are not strictly separated")


def _normalized_raw(name: str, values: pd.Series) -> np.ndarray:
    x = pd.to_numeric(values, errors="raise").to_numpy(dtype=np.float64, copy=False)
    if name == "f_4":
        out = np.log1p(np.maximum(x, 0.0)) / 25.0
    elif name == "f_3":
        out = x / 32.0
    else:
        out = np.clip((x - 1.0) / 0.10, -4.0, 4.0)
    return out.astype(np.float32)


def build_base_features(
    train: pd.DataFrame, test: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray, list[str], np.ndarray, np.ndarray]:
    """Build target-free features with causal lags across the train/test boundary."""
    n_train = len(train)
    all_rows = pd.concat(
        [train[KEY_COLUMNS + RAW_FEATURES], test[KEY_COLUMNS + RAW_FEATURES]],
        axis=0,
        ignore_index=True,
    )
    # Kaggle stores f_3 as numeric strings in train and integers in test.
    # Normalize before ranking so comparisons are well-defined across frames.
    all_rows["f_3"] = pd.to_numeric(all_rows["f_3"], errors="raise").astype(np.int16)
    if not all_rows["date"].is_monotonic_increasing:
        raise ValueError("rows must be globally ordered by date for causal group shifts")

    features: dict[str, np.ndarray] = {}
    for column in RAW_FEATURES:
        features[f"raw_{column}"] = _normalized_raw(column, all_rows[column])

    rank_names: list[str] = []
    for column in RAW_FEATURES:
        name = f"cs_rank_{column}"
        ranked = all_rows.groupby("date", sort=False, observed=True)[column].rank(
            method="average", pct=True
        )
        features[name] = (ranked.to_numpy(dtype=np.float64) - 0.5).astype(np.float32)
        rank_names.append(name)

    date_values = all_rows["date"].to_numpy(dtype=np.float64)
    features["date_scaled"] = ((date_values - date_values.min()) / (date_values.max() - date_values.min()) - 0.5).astype(np.float32)

    # The concatenation is chronological, so groupby.shift is causal and seeds test
    # lags from the training tail for overlapping stock codes.
    codes = all_rows["code"]
    lag_sources = [f"cs_rank_f_{i}" for i in (0, 1, 2, 5, 6)]
    for source in lag_sources:
        series = pd.Series(features[source], copy=False)
        shifted_1 = series.groupby(codes, sort=False, observed=True).shift(1)
        shifted_5 = series.groupby(codes, sort=False, observed=True).shift(5)
        lag_1 = shifted_1.fillna(0.0).to_numpy(dtype=np.float32)
        lag_5 = shifted_5.fillna(0.0).to_numpy(dtype=np.float32)
        features[f"{source}_lag1"] = lag_1
        features[f"{source}_lag5"] = lag_5
        features[f"{source}_diff1"] = features[source] - lag_1

    feature_names = list(features)
    matrix = np.column_stack([features[name] for name in feature_names]).astype(np.float32, copy=False)
    if not np.isfinite(matrix).all():
        raise ValueError("engineered feature matrix contains non-finite values")

    all_code_ids, _ = pd.factorize(all_rows["code"], sort=True)
    del all_rows, features
    gc.collect()
    return (
        matrix[:n_train],
        matrix[n_train:],
        feature_names,
        all_code_ids[:n_train].astype(np.int32),
        all_code_ids[n_train:].astype(np.int32),
    )


def within_date_rank(values: np.ndarray, dates: np.ndarray) -> np.ndarray:
    frame = pd.DataFrame({"date": dates, "value": values})
    ranked = frame.groupby("date", sort=False, observed=True)["value"].rank(
        method="average", pct=True
    )
    return (ranked.to_numpy(dtype=np.float64) - 0.5).astype(np.float32)


def daily_rank_ic(y_true: np.ndarray, y_pred: np.ndarray, dates: np.ndarray) -> dict[str, Any]:
    true_rank = within_date_rank(y_true, dates).astype(np.float64)
    pred_rank = within_date_rank(y_pred, dates).astype(np.float64)
    frame = pd.DataFrame(
        {
            "date": dates,
            "x": pred_rank,
            "y": true_rank,
            "xy": pred_rank * true_rank,
            "x2": pred_rank * pred_rank,
            "y2": true_rank * true_rank,
        }
    )
    stats = frame.groupby("date", sort=False, observed=True).agg(
        x=("x", "mean"),
        y=("y", "mean"),
        xy=("xy", "mean"),
        x2=("x2", "mean"),
        y2=("y2", "mean"),
        n=("x", "size"),
    )
    numerator = stats["xy"] - stats["x"] * stats["y"]
    denominator = np.sqrt(
        (stats["x2"] - stats["x"] ** 2) * (stats["y2"] - stats["y"] ** 2)
    )
    values = (numerator / denominator.replace(0.0, np.nan)).dropna()
    return {
        "mean": float(values.mean()),
        "std": float(values.std(ddof=0)),
        "median": float(values.median()),
        "min": float(values.min()),
        "max": float(values.max()),
        "dates": int(len(values)),
        "rows": int(len(frame)),
    }


def build_expanding_code_prior(
    y_rank: np.ndarray, code_ids: np.ndarray, smoothing: float = 20.0
) -> tuple[np.ndarray, np.ndarray]:
    frame = pd.DataFrame({"code": code_ids, "target": y_rank})
    previous_count = frame.groupby("code", sort=False, observed=True).cumcount().to_numpy(dtype=np.float32)
    cumulative = frame.groupby("code", sort=False, observed=True)["target"].cumsum().to_numpy(dtype=np.float64)
    previous_sum = cumulative - y_rank
    denominator = previous_count + smoothing
    prior = np.divide(
        previous_sum,
        denominator,
        out=np.zeros_like(previous_sum, dtype=np.float64),
        where=denominator > 0,
    )
    coverage = np.log1p(previous_count) / 10.0
    return prior.astype(np.float32), coverage.astype(np.float32)


def frozen_code_prior(
    y_rank: np.ndarray,
    code_ids: np.ndarray,
    fit_mask: np.ndarray,
    apply_mask: np.ndarray,
    smoothing: float = 20.0,
) -> tuple[np.ndarray, np.ndarray]:
    n_codes = int(code_ids.max()) + 1
    sums = np.bincount(code_ids[fit_mask], weights=y_rank[fit_mask], minlength=n_codes)
    counts = np.bincount(code_ids[fit_mask], minlength=n_codes)
    apply_codes = code_ids[apply_mask]
    prior = sums[apply_codes] / (counts[apply_codes] + smoothing)
    coverage = np.log1p(counts[apply_codes]) / 10.0
    return prior.astype(np.float32), coverage.astype(np.float32)


def add_prior_columns(base: np.ndarray, prior: np.ndarray, coverage: np.ndarray) -> np.ndarray:
    return np.column_stack((base, prior, coverage)).astype(np.float32, copy=False)


def mean_daily_prediction_correlation(a: np.ndarray, b: np.ndarray, dates: np.ndarray) -> float:
    return float(daily_rank_ic(a, b, dates)["mean"])


def fit_fold(
    fold: FoldSpec,
    base_train: np.ndarray,
    y_rank: np.ndarray,
    dates: np.ndarray,
    code_ids: np.ndarray,
    expanding_prior: np.ndarray,
    expanding_coverage: np.ndarray,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    train_mask = dates <= fold.train_end
    valid_mask = (dates >= fold.valid_start) & (dates <= fold.valid_end)
    if int(dates[train_mask].max()) >= int(dates[valid_mask].min()):
        raise AssertionError("temporal fold overlap")

    valid_prior, valid_coverage = frozen_code_prior(
        y_rank, code_ids, train_mask, valid_mask
    )
    x_train = add_prior_columns(
        base_train[train_mask], expanding_prior[train_mask], expanding_coverage[train_mask]
    )
    x_valid = add_prior_columns(base_train[valid_mask], valid_prior, valid_coverage)
    y_train = y_rank[train_mask]
    y_valid = y_rank[valid_mask]
    valid_dates = dates[valid_mask]

    ridge = Ridge(alpha=1000.0, fit_intercept=True, solver="lsqr", tol=1e-4)
    ridge.fit(x_train, y_train)
    ridge_pred = ridge.predict(x_valid).astype(np.float32)

    params = {
        "objective": "regression",
        "metric": "l2",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "max_depth": -1,
        "min_data_in_leaf": 2000,
        "feature_fraction": 0.85,
        "bagging_fraction": 0.85,
        "bagging_freq": 1,
        "lambda_l1": 0.2,
        "lambda_l2": 2.0,
        "max_bin": 127,
        "seed": SEED,
        "feature_fraction_seed": SEED,
        "bagging_seed": SEED,
        "num_threads": min(16, os.cpu_count() or 4),
        "verbosity": -1,
        "force_col_wise": True,
    }
    train_set = lgb.Dataset(x_train, label=y_train, free_raw_data=False)
    valid_set = lgb.Dataset(x_valid, label=y_valid, reference=train_set, free_raw_data=False)
    booster = lgb.train(
        params,
        train_set,
        num_boost_round=300,
        valid_sets=[valid_set],
        valid_names=["future_dates"],
        callbacks=[lgb.early_stopping(35, verbose=False), lgb.log_evaluation(50)],
    )
    tree_pred = booster.predict(x_valid, num_iteration=booster.best_iteration).astype(np.float32)

    predictions = {
        "prior": valid_prior,
        "ridge": ridge_pred,
        "lightgbm": tree_pred,
    }
    metrics = {name: daily_rank_ic(y_valid, pred, valid_dates) for name, pred in predictions.items()}
    correlations = {
        "ridge_lightgbm": mean_daily_prediction_correlation(ridge_pred, tree_pred, valid_dates),
        "prior_lightgbm": mean_daily_prediction_correlation(valid_prior, tree_pred, valid_dates),
        "prior_ridge": mean_daily_prediction_correlation(valid_prior, ridge_pred, valid_dates),
    }
    receipt = {
        "fold": asdict(fold),
        "train_rows": int(train_mask.sum()),
        "valid_rows": int(valid_mask.sum()),
        "train_date_range": [int(dates[train_mask].min()), int(dates[train_mask].max())],
        "valid_date_range": [int(valid_dates.min()), int(valid_dates.max())],
        "strict_temporal_order": bool(dates[train_mask].max() < valid_dates.min()),
        "best_iteration": int(booster.best_iteration),
        "metrics": metrics,
        "prediction_rank_ic_correlations": correlations,
    }
    del x_train, x_valid, y_train, train_set, valid_set, ridge, booster
    gc.collect()
    return receipt, {
        **predictions,
        "target": y_valid,
        "dates": valid_dates,
    }


def select_blend(oof: dict[str, np.ndarray]) -> tuple[dict[str, float], dict[str, Any], list[dict[str, Any]]]:
    dates = oof["dates"]
    target = oof["target"]
    members = {
        name: within_date_rank(oof[name], dates)
        for name in ("prior", "ridge", "lightgbm")
    }
    candidates: list[dict[str, Any]] = []
    for prior_units in range(0, 11):
        for ridge_units in range(0, 11 - prior_units):
            tree_units = 10 - prior_units - ridge_units
            weights = {
                "prior": prior_units / 10.0,
                "ridge": ridge_units / 10.0,
                "lightgbm": tree_units / 10.0,
            }
            pred = sum(weights[name] * members[name] for name in weights)
            metric = daily_rank_ic(target, pred, dates)
            candidates.append({"weights": weights, "metric": metric})
    candidates.sort(key=lambda item: item["metric"]["mean"], reverse=True)
    best = candidates[0]
    return best["weights"], best["metric"], candidates[:10]


def fit_final_models(
    base_train: np.ndarray,
    base_test: np.ndarray,
    y_rank: np.ndarray,
    train_code_ids: np.ndarray,
    test_code_ids: np.ndarray,
    expanding_prior: np.ndarray,
    expanding_coverage: np.ndarray,
    best_iterations: Iterable[int],
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    train_mask = np.ones(len(y_rank), dtype=bool)
    test_mask = np.ones(len(test_code_ids), dtype=bool)
    n_codes = max(int(train_code_ids.max()), int(test_code_ids.max())) + 1
    sums = np.bincount(train_code_ids, weights=y_rank, minlength=n_codes)
    counts = np.bincount(train_code_ids, minlength=n_codes)
    test_prior = (sums[test_code_ids] / (counts[test_code_ids] + 20.0)).astype(np.float32)
    test_coverage = (np.log1p(counts[test_code_ids]) / 10.0).astype(np.float32)

    x_train = add_prior_columns(base_train, expanding_prior, expanding_coverage)
    x_test = add_prior_columns(base_test, test_prior, test_coverage)

    ridge = Ridge(alpha=1000.0, fit_intercept=True, solver="lsqr", tol=1e-4)
    ridge.fit(x_train, y_rank)
    ridge_pred = ridge.predict(x_test).astype(np.float32)

    final_iterations = max(50, int(round(float(np.median(list(best_iterations))))))
    params = {
        "objective": "regression",
        "metric": "l2",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "min_data_in_leaf": 2000,
        "feature_fraction": 0.85,
        "bagging_fraction": 0.85,
        "bagging_freq": 1,
        "lambda_l1": 0.2,
        "lambda_l2": 2.0,
        "max_bin": 127,
        "seed": SEED,
        "feature_fraction_seed": SEED,
        "bagging_seed": SEED,
        "num_threads": min(16, os.cpu_count() or 4),
        "verbosity": -1,
        "force_col_wise": True,
    }
    train_set = lgb.Dataset(x_train, label=y_rank, free_raw_data=False)
    booster = lgb.train(params, train_set, num_boost_round=final_iterations)
    tree_pred = booster.predict(x_test, num_iteration=final_iterations).astype(np.float32)

    diagnostics = {
        "iterations": final_iterations,
        "test_prior_known_code_fraction": float(np.mean(counts[test_code_ids] > 0)),
        "test_prior_count_percentiles": {
            str(q): float(np.quantile(counts[test_code_ids], q))
            for q in (0.0, 0.25, 0.5, 0.75, 1.0)
        },
    }
    del x_train, x_test, train_set, ridge, booster, train_mask, test_mask
    gc.collect()
    return {"prior": test_prior, "ridge": ridge_pred, "lightgbm": tree_pred}, diagnostics


def audit_submission(
    submission: pd.DataFrame, sample: pd.DataFrame, test: pd.DataFrame
) -> dict[str, Any]:
    checks = {
        "columns_exact": list(submission.columns) == list(sample.columns),
        "row_count_exact": len(submission) == len(sample) == len(test),
        "ids_exact": submission["id"].equals(sample["id"]),
        "keys_exact": submission[KEY_COLUMNS].equals(sample[KEY_COLUMNS]),
        "ids_unique": bool(submission["id"].is_unique),
        "keys_unique": bool(not submission.duplicated(KEY_COLUMNS).any()),
        "prediction_numeric": bool(pd.api.types.is_numeric_dtype(submission[PREDICTION])),
        "prediction_finite": bool(np.isfinite(submission[PREDICTION].to_numpy()).all()),
        "prediction_nonconstant": bool(submission[PREDICTION].nunique() > 1),
    }
    if not all(checks.values()):
        raise ValueError(f"submission audit failed: {checks}")
    return {
        "checks": checks,
        "prediction_summary": {
            key: float(value)
            for key, value in submission[PREDICTION].describe().to_dict().items()
        },
    }


def write_submission_zip(submission: pd.DataFrame, csv_path: Path, zip_path: Path) -> dict[str, Any]:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(csv_path, index=False)
    csv_sha = sha256_file(csv_path)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        archive.write(csv_path, arcname="submission.csv")
    with zipfile.ZipFile(zip_path, "r") as archive:
        bad_member = archive.testzip()
        members = archive.namelist()
    if bad_member is not None or members != ["submission.csv"]:
        raise ValueError(f"invalid submission ZIP: bad={bad_member}, members={members}")
    result = {
        "csv_path": str(csv_path),
        "csv_bytes": csv_path.stat().st_size,
        "csv_sha256": csv_sha,
        "zip_path": str(zip_path),
        "zip_bytes": zip_path.stat().st_size,
        "zip_sha256": sha256_file(zip_path),
        "zip_crc_ok": True,
        "zip_members": members,
    }
    # The ZIP is the submission artifact; discard its reproducible uncompressed temporary.
    csv_path.unlink()
    result["csv_removed_after_verified_zip"] = True
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--artifacts-dir", type=Path, default=Path("artifacts"))
    args = parser.parse_args()
    started = time.perf_counter()

    paths = {
        "train": args.data_dir / "train_data.pkl",
        "test": args.data_dir / "test_data.pkl",
        "sample": args.data_dir / "sample_submission.csv",
    }
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)

    print("Loading V2 competition data...", flush=True)
    train = pd.read_pickle(paths["train"])
    test = pd.read_pickle(paths["test"])
    sample = pd.read_csv(paths["sample"])
    assert_input_contract(train, test, sample)

    input_receipt = {
        name: {
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for name, path in paths.items()
    }
    host_reference = {
        "source": "official sample_submission.csv",
        "treated_as_model_evidence": False,
        "nonconstant": bool(sample[PREDICTION].nunique() > 1),
        "summary": {
            key: float(value)
            for key, value in sample[PREDICTION].describe().to_dict().items()
        },
        "warning": "Unscored official-file baseline; leaderboard implications are inference, not proof.",
    }

    print("Building target-free cross-sectional and causal lag features...", flush=True)
    base_train, base_test, feature_names, train_code_ids, test_code_ids = build_base_features(train, test)
    dates = train["date"].to_numpy(dtype=np.int32)
    test_dates = test["date"].to_numpy(dtype=np.int32)
    y_rank = within_date_rank(train[TARGET].to_numpy(dtype=np.float64), dates)
    expanding_prior, expanding_coverage = build_expanding_code_prior(y_rank, train_code_ids)

    fold_receipts: list[dict[str, Any]] = []
    oof_parts: dict[str, list[np.ndarray]] = {
        key: [] for key in ("prior", "ridge", "lightgbm", "target", "dates")
    }
    for fold in FOLDS:
        print(f"Training {fold.name}...", flush=True)
        fold_receipt, fold_predictions = fit_fold(
            fold,
            base_train,
            y_rank,
            dates,
            train_code_ids,
            expanding_prior,
            expanding_coverage,
        )
        fold_receipts.append(fold_receipt)
        for key in oof_parts:
            oof_parts[key].append(fold_predictions[key])
        print(json.dumps(fold_receipt["metrics"], sort_keys=True), flush=True)

    oof = {key: np.concatenate(parts) for key, parts in oof_parts.items()}
    blend_weights, blend_metric, top_blends = select_blend(oof)
    print(f"Selected OOF blend {blend_weights}: {blend_metric['mean']:.6f}", flush=True)

    print("Fitting full-history models and predicting test dates...", flush=True)
    final_members, final_diagnostics = fit_final_models(
        base_train,
        base_test,
        y_rank,
        train_code_ids,
        test_code_ids,
        expanding_prior,
        expanding_coverage,
        [item["best_iteration"] for item in fold_receipts],
    )
    ranked_members = {
        name: within_date_rank(values, test_dates)
        for name, values in final_members.items()
    }
    blended = sum(blend_weights[name] * ranked_members[name] for name in blend_weights)
    final_prediction = within_date_rank(blended, test_dates)

    prediction_frame = test[KEY_COLUMNS].copy()
    prediction_frame[PREDICTION] = final_prediction
    submission = sample.drop(columns=[PREDICTION]).merge(
        prediction_frame, on=KEY_COLUMNS, how="left", validate="one_to_one"
    )
    submission = submission[list(sample.columns)]
    submission_audit = audit_submission(submission, sample, test)

    submissions_dir = args.artifacts_dir / "submissions"
    zip_receipt = write_submission_zip(
        submission,
        submissions_dir / "stocks_v2_model_blend_submission.csv",
        submissions_dir / "stocks_v2_model_blend_submission.zip",
    )
    host_audit = audit_submission(sample.copy(), sample, test)

    receipt = {
        "schema_version": "stocks-v2-run-receipt-v1",
        "competition": COMPETITION,
        "created_at_utc": utc_now(),
        "deadline_utc": "2026-07-15T16:00:00+00:00",
        "external_submission_executed": False,
        "input_receipt": input_receipt,
        "data_contract": {
            "train_shape": list(train.shape),
            "test_shape": list(test.shape),
            "sample_shape": list(sample.shape),
            "train_dates": [int(train.date.min()), int(train.date.max())],
            "test_dates": [int(test.date.min()), int(test.date.max())],
            "train_codes": int(train.code.nunique()),
            "test_codes": int(test.code.nunique()),
            "overlapping_codes": int(len(set(train.code) & set(test.code))),
            "train_test_dates_strictly_separated": bool(train.date.max() < test.date.min()),
        },
        "metric_contract": {
            "name": "mean daily Rank IC",
            "implementation": "mean across dates of Spearman rank correlation",
            "direction": "maximize",
        },
        "validation_contract": {
            "type": "expanding temporal holdout",
            "target_encoding": "past-only expanding train prior and fold-frozen validation prior",
            "same_date_cross_sectional_features": True,
            "feature_lags": "current or previous feature rows only; no target lags",
        },
        "features": feature_names + ["past_code_target_prior", "past_code_observation_coverage"],
        "feature_count": len(feature_names) + 2,
        "folds": fold_receipts,
        "oof": {
            "rows": int(len(oof["target"])),
            "dates": int(np.unique(oof["dates"]).size),
            "selected_blend_weights": blend_weights,
            "selected_blend_metric": blend_metric,
            "top_10_blends": top_blends,
        },
        "final_fit": final_diagnostics,
        "submission_audit": submission_audit,
        "submission_artifact": zip_receipt,
        "host_reference": host_reference,
        "host_reference_audit": host_audit,
        "recommended_probe_order": [
            {
                "artifact": str(paths["sample"]),
                "kind": "official host reference",
                "reason": "Nonconstant official predictions and near-perfect leaderboard entries make this the cheapest diagnostic probe; score is unverified.",
            },
            {
                "artifact": zip_receipt["zip_path"],
                "kind": "leakage-safe trained blend",
                "reason": "Independent model candidate with expanding-date OOF evidence.",
            },
        ],
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__,
            "lightgbm": lgb.__version__,
            "seed": SEED,
        },
        "runtime_seconds": float(time.perf_counter() - started),
    }
    receipt_path = args.artifacts_dir / "receipts" / "stocks_v2_run.json"
    atomic_json(receipt_path, receipt)

    handoff = {
        "competition": COMPETITION,
        "external_submission_executed": False,
        "host_reference_command": (
            "kaggle competitions submit -c stocks-return-prediction-v-2 "
            f"-f {paths['sample']} -m \"official host reference diagnostic; unchanged file\""
        ),
        "model_command": (
            "kaggle competitions submit -c stocks-return-prediction-v-2 "
            f"-f {zip_receipt['zip_path']} -m \"expanding-time CV rank blend v1\""
        ),
        "receipt": str(receipt_path),
        "model_zip_sha256": zip_receipt["zip_sha256"],
    }
    atomic_json(args.artifacts_dir / "SUBMISSION_HANDOFF.json", handoff)
    print(json.dumps(handoff, indent=2), flush=True)


if __name__ == "__main__":
    main()
