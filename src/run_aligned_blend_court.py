#!/usr/bin/env python3
"""Aligned, leakage-safe OOF court for v1 plus causal-distilled Stocks V2.

Blend weight is selected on the earlier future fold only. The later fold stays
sealed until selection. A local ZIP is built only when the chosen blend improves
both folds, mean score, and worst-fold score over v1 under predeclared margins.
This script never submits to Kaggle.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import sys
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd

from run_causal_distilled import FOLDS as DISTILLED_FOLDS
from run_causal_distilled import build_causal_features
from run_formula_candidate import exact_date_log_product
from run_pipeline import (
    KEY_COLUMNS,
    PREDICTION,
    RAW_FEATURES,
    FoldSpec,
    audit_submission,
    build_base_features,
    build_expanding_code_prior,
    daily_rank_ic,
    fit_fold,
    sha256_file,
    within_date_rank,
    write_submission_zip,
)


SEED = 20260712
V1_WEIGHT = 0.8
WEIGHT_GRID = tuple(round(v, 2) for v in np.arange(0.0, 1.01, 0.1))
MIN_MEAN_MARGIN = 0.0005
MIN_WORST_MARGIN = 0.0002
V1_ZIP_SHA = "da532638f8de48ae24a5d5d45a386621d1f617c614f17c6f63772cdf8184a56a"
DISTILLED_ZIP_SHA = "7631c79b81f7926d803aa80dc4fc8da2a6ea27f562701e48e397dbe57069e215"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def v1_oof(train: pd.DataFrame) -> list[dict[str, Any]]:
    empty = train[KEY_COLUMNS + RAW_FEATURES].iloc[:0].copy()
    base, _, _, code_ids, _ = build_base_features(train, empty)
    # Match the deployed v1's static train+test date scaling without allocating
    # the full test feature matrix. All other training features are unchanged.
    date_col = 14
    base[:, date_col] = train["date"].to_numpy(dtype=np.float32) / 2803.0 - 0.5
    dates = train["date"].to_numpy(dtype=np.int32)
    y_rank = within_date_rank(train["y"].to_numpy(dtype=np.float64), dates)
    expanding_prior, expanding_coverage = build_expanding_code_prior(y_rank, code_ids)
    specs = (
        FoldSpec("aligned_v1_1201_1450", 1200, 1201, 1450),
        FoldSpec("aligned_v1_1451_1696", 1450, 1451, 1696),
    )
    result = []
    for spec in specs:
        receipt, predictions = fit_fold(
            spec,
            base,
            y_rank,
            dates,
            code_ids,
            expanding_prior,
            expanding_coverage,
        )
        valid_dates = predictions["dates"]
        tree = within_date_rank(predictions["lightgbm"], valid_dates)
        ridge = within_date_rank(predictions["ridge"], valid_dates)
        blend = V1_WEIGHT * tree + (1.0 - V1_WEIGHT) * ridge
        result.append(
            {
                "receipt": receipt,
                "prediction": blend.astype(np.float32),
                "target": predictions["target"].astype(np.float32),
                "dates": valid_dates.astype(np.int32),
            }
        )
    del base, expanding_prior, expanding_coverage
    gc.collect()
    return result


def distilled_oof(train: pd.DataFrame) -> list[dict[str, Any]]:
    dates = train["date"].to_numpy(dtype=np.int32)
    official = train["y"].to_numpy(dtype=np.float64)
    formula, complete = exact_date_log_product(train)
    label = within_date_rank(np.where(complete, formula, 0.0), dates)
    features, _ = build_causal_features(train)
    result = []
    for fold in DISTILLED_FOLDS:
        train_mask = (dates <= fold.purged_label_end) & complete
        valid_mask = (dates >= fold.valid_start) & (dates <= fold.valid_end)
        valid_formula = valid_mask & complete
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
        train_set = lgb.Dataset(features[train_mask], label=label[train_mask], free_raw_data=False)
        valid_set = lgb.Dataset(
            features[valid_formula], label=label[valid_formula], reference=train_set, free_raw_data=False
        )
        booster = lgb.train(
            params,
            train_set,
            num_boost_round=180,
            valid_sets=[valid_set],
            callbacks=[lgb.early_stopping(25, verbose=False), lgb.log_evaluation(0)],
        )
        prediction = booster.predict(features[valid_mask], num_iteration=booster.best_iteration)
        result.append(
            {
                "fold": fold.name,
                "best_iteration": int(booster.best_iteration),
                "prediction": prediction.astype(np.float32),
                "target": official[valid_mask].astype(np.float32),
                "dates": dates[valid_mask].astype(np.int32),
            }
        )
        del train_set, valid_set, booster
        gc.collect()
    del features
    gc.collect()
    return result


def prediction_zip(path: Path) -> pd.DataFrame:
    with zipfile.ZipFile(path) as archive:
        if archive.testzip() is not None or archive.namelist() != ["submission.csv"]:
            raise ValueError(f"invalid candidate ZIP {path}")
        with archive.open("submission.csv") as handle:
            return pd.read_csv(handle, usecols=KEY_COLUMNS + [PREDICTION])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--artifacts-dir", type=Path, default=Path("artifacts"))
    args = parser.parse_args()
    started = time.perf_counter()
    train = pd.read_pickle(args.data_dir / "train_data.pkl")
    print("Reconstructing aligned v1 OOF...", flush=True)
    v1 = v1_oof(train)
    print("Reconstructing aligned causal-distilled OOF...", flush=True)
    distilled = distilled_oof(train)

    fold_rows = []
    grid_rows = []
    for i, (a, b) in enumerate(zip(v1, distilled)):
        if not np.array_equal(a["dates"], b["dates"]):
            raise AssertionError(f"fold {i}: OOF date rows are not aligned")
        # v1 stores the within-date-ranked target while distilled retains raw y;
        # they are metric-equivalent, not numerically equal.
        target_alignment = daily_rank_ic(a["target"], b["target"], a["dates"])["mean"]
        if target_alignment < 0.999999:
            raise AssertionError(f"fold {i}: OOF targets are not rank-aligned ({target_alignment})")
        a_rank = within_date_rank(a["prediction"], a["dates"])
        b_rank = within_date_rank(b["prediction"], b["dates"])
        base_metric = daily_rank_ic(a["target"], a_rank, a["dates"])
        fold_rows.append(
            {
                "fold": i + 1,
                "dates": [int(a["dates"].min()), int(a["dates"].max())],
                "v1": base_metric,
                "distilled": daily_rank_ic(a["target"], b_rank, a["dates"]),
                "prediction_rank_correlation": daily_rank_ic(a_rank, b_rank, a["dates"])["mean"],
            }
        )
        for weight in WEIGHT_GRID:
            pred = (1.0 - weight) * a_rank + weight * b_rank
            grid_rows.append(
                {
                    "fold": i + 1,
                    "distilled_weight": weight,
                    "metric": daily_rank_ic(a["target"], pred, a["dates"])["mean"],
                }
            )

    # Select using fold 1 only. Fold 2 is sealed until this point.
    early = [row for row in grid_rows if row["fold"] == 1]
    selected_weight = max(early, key=lambda row: (row["metric"], -row["distilled_weight"]))[
        "distilled_weight"
    ]
    selected_scores = [
        next(row["metric"] for row in grid_rows if row["fold"] == fold and row["distilled_weight"] == selected_weight)
        for fold in (1, 2)
    ]
    v1_scores = [row["v1"]["mean"] for row in fold_rows]
    gate = {
        "weight_selected_on_fold1_only": selected_weight,
        "fold_scores_v1": v1_scores,
        "fold_scores_selected_blend": selected_scores,
        "each_fold_strictly_better": bool(all(x > y for x, y in zip(selected_scores, v1_scores))),
        "mean_margin": float(np.mean(selected_scores) - np.mean(v1_scores)),
        "worst_fold_margin": float(min(selected_scores) - min(v1_scores)),
        "minimum_mean_margin": MIN_MEAN_MARGIN,
        "minimum_worst_fold_margin": MIN_WORST_MARGIN,
    }
    gate["passed"] = bool(
        gate["each_fold_strictly_better"]
        and gate["mean_margin"] >= MIN_MEAN_MARGIN
        and gate["worst_fold_margin"] >= MIN_WORST_MARGIN
        and selected_weight > 0.0
    )

    artifact = None
    audit = None
    if gate["passed"]:
        print(f"Blend gate passed at distilled weight={selected_weight:.2f}; building local ZIP...", flush=True)
        test = pd.read_pickle(args.data_dir / "test_data.pkl")
        sample = pd.read_csv(args.data_dir / "sample_submission.csv")
        v1_zip = args.artifacts_dir / "submissions" / "stocks_v2_model_blend_submission.zip"
        distilled_zip = args.artifacts_dir / "submissions" / "stocks_v2_causal_distilled_submission.zip"
        if sha256_file(v1_zip) != V1_ZIP_SHA or sha256_file(distilled_zip) != DISTILLED_ZIP_SHA:
            raise ValueError("parent candidate hash drift")
        v1_test = prediction_zip(v1_zip).rename(columns={PREDICTION: "v1"})
        d_test = prediction_zip(distilled_zip).rename(columns={PREDICTION: "distilled"})
        aligned = test[KEY_COLUMNS].merge(v1_test, on=KEY_COLUMNS, validate="one_to_one").merge(
            d_test, on=KEY_COLUMNS, validate="one_to_one"
        )
        dates = aligned["date"].to_numpy(dtype=np.int32)
        blended = (1.0 - selected_weight) * within_date_rank(aligned["v1"].to_numpy(), dates)
        blended += selected_weight * within_date_rank(aligned["distilled"].to_numpy(), dates)
        prediction = within_date_rank(blended, dates)
        prediction_frame = aligned[KEY_COLUMNS].copy()
        prediction_frame[PREDICTION] = prediction
        submission = sample.drop(columns=[PREDICTION]).merge(
            prediction_frame, on=KEY_COLUMNS, how="left", validate="one_to_one"
        )[list(sample.columns)]
        audit = audit_submission(submission, sample, test)
        artifact = write_submission_zip(
            submission,
            args.artifacts_dir / "submissions" / "stocks_v2_v1_distilled_blend_submission.csv",
            args.artifacts_dir / "submissions" / "stocks_v2_v1_distilled_blend_submission.zip",
        )

    receipt = {
        "schema_version": "stocks-v2-aligned-blend-court-v1",
        "competition": "stocks-return-prediction-v-2",
        "created_at_utc": utc_now(),
        "external_submission_executed": False,
        "actual_public_scores": {"v1": 0.09057, "causal_distilled": 0.08961},
        "selection_contract": {
            "weight_grid": list(WEIGHT_GRID),
            "selection_fold": 1,
            "sealed_confirmation_fold": 2,
            "build_only_if_gate_passes": True,
        },
        "folds": fold_rows,
        "weight_grid_results": grid_rows,
        "gate": gate,
        "decision": "BUILD_LOCAL_BLEND_NO_SUBMIT" if gate["passed"] else "KILL_BLEND_NO_ARTIFACT",
        "submission_audit": audit,
        "submission_artifact": artifact,
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
    path = args.artifacts_dir / "receipts" / "stocks_v2_aligned_blend_court.json"
    path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"receipt": str(path), "decision": receipt["decision"], "gate": gate, "artifact": artifact}, indent=2), flush=True)


if __name__ == "__main__":
    main()
