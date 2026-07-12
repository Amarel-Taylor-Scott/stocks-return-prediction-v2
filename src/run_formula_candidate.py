#!/usr/bin/env python3
"""Build and audit the Stocks V2 exact-date future-return candidate.

This is a static-batch formula reconstruction, not an online forecasting model.
Train-only diagnosis shows that the within-date rank of ``y[code,date]`` is
almost exactly the rank of the four-session compounded ``f_2`` move at exact
dates ``date+2`` through ``date+5``.  The competition provides all test feature
rows in one static file, so the formula is computable for all but the final five
dates and sparse code/date holes.

The script never submits.  The receipt keeps the compliance ambiguity explicit:
the official page says to predict with the feature data in the test set and does
not specify sequential serving, but the task is described as future-return
prediction and the only specific rule says "Don't cheat!".  Human/organizer
review is required before any external submission.
"""

from __future__ import annotations

import argparse
import gc
import json
import platform
import sys
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from run_pipeline import (
    KEY_COLUMNS,
    PREDICTION,
    audit_submission,
    atomic_json,
    daily_rank_ic,
    sha256_file,
    within_date_rank,
    write_submission_zip,
)


COMPETITION = "stocks-return-prediction-v-2"
BASELINE_OOF = 0.0894160876163553
OFFSETS = (2, 3, 4, 5)
FOLDS = (("future_1201_1450", 1201, 1450), ("future_1451_1696", 1451, 1696))


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def exact_date_log_product(
    frame: pd.DataFrame,
    offsets: Iterable[int] = OFFSETS,
    feature: str = "f_2",
) -> tuple[np.ndarray, np.ndarray]:
    """Sum log(feature[code,date+k]) on an exact dense code/date grid.

    This intentionally does not use groupby.shift: when a code is missing on a
    date, "next observed row" is not the requested exact date and loses roughly
    0.005 Rank IC on the train-only court.
    """

    offsets = tuple(int(k) for k in offsets)
    if not offsets:
        raise ValueError("offsets must be nonempty")
    if frame.duplicated(KEY_COLUMNS).any():
        raise ValueError("duplicate (code,date) keys")
    dates = pd.to_numeric(frame["date"], errors="raise").to_numpy(dtype=np.int32)
    if dates.min() < 0:
        raise ValueError("negative dates are unsupported")
    values = pd.to_numeric(frame[feature], errors="raise").to_numpy(dtype=np.float64)
    if not np.isfinite(values).all() or np.any(values <= 0):
        raise ValueError(f"{feature} must be finite and strictly positive")
    code_ids, _ = pd.factorize(frame["code"], sort=True)
    min_offset, max_offset = min(offsets), max(offsets)
    left_pad = max(0, -min_offset)
    width = int(dates.max()) + left_pad + max(0, max_offset) + 1
    grid = np.full((int(code_ids.max()) + 1, width), np.nan, dtype=np.float32)
    grid[code_ids, dates + left_pad] = np.log(values).astype(np.float32)
    score = np.zeros(len(frame), dtype=np.float32)
    complete = np.ones(len(frame), dtype=bool)
    for offset in offsets:
        index = dates + left_pad + offset
        in_bounds = (index >= 0) & (index < width)
        safe_index = np.clip(index, 0, width - 1)
        part = grid[code_ids, safe_index]
        present = in_bounds & np.isfinite(part)
        score += np.where(present, part, 0.0).astype(np.float32)
        complete &= present
    del grid
    gc.collect()
    return score, complete


def rotate_within_date(values: np.ndarray, dates: np.ndarray) -> np.ndarray:
    """Distribution-matched code-alignment placebo: rotate one row per date."""

    result = np.empty_like(values)
    boundaries = np.r_[0, np.flatnonzero(dates[1:] != dates[:-1]) + 1, len(dates)]
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        result[start:end] = np.roll(values[start:end], 1)
    return result


def metric_on_dates(
    target: np.ndarray,
    prediction: np.ndarray,
    dates: np.ndarray,
    start: int,
    end: int,
    eligible: np.ndarray | None = None,
) -> dict[str, Any]:
    mask = (dates >= start) & (dates <= end)
    if eligible is not None:
        mask &= eligible
    return daily_rank_ic(target[mask], prediction[mask], dates[mask])


def validate_formula(train: pd.DataFrame) -> dict[str, Any]:
    dates = train["date"].to_numpy(dtype=np.int32)
    target = train["y"].to_numpy(dtype=np.float64)
    exact, complete = exact_date_log_product(train)
    # A neutral score for sparse code/date holes is empirically safer than
    # skipping to the next observation, and it lets the court score every row.
    candidate = np.where(complete, exact, 0.0).astype(np.float32)

    wrong_left, wrong_left_ok = exact_date_log_product(train, (1, 2, 3, 4))
    wrong_right, wrong_right_ok = exact_date_log_product(train, (3, 4, 5, 6))
    causal, causal_ok = exact_date_log_product(train, (-5, -4, -3, -2))
    rotated = rotate_within_date(candidate, dates)

    folds = []
    for name, start, end in FOLDS:
        metric = metric_on_dates(target, candidate, dates, start, end)
        exact_only = metric_on_dates(target, exact, dates, start, end, complete)
        fold_mask = (dates >= start) & (dates <= end)
        placebos = {
            "causal_mirror_exact_dates": metric_on_dates(
                target, causal, dates, start, end, causal_ok
            ),
            "wrong_window_date_plus_1_to_4": metric_on_dates(
                target, wrong_left, dates, start, end, wrong_left_ok
            ),
            "wrong_window_date_plus_3_to_6": metric_on_dates(
                target, wrong_right, dates, start, end, wrong_right_ok
            ),
            "within_date_code_rotation": metric_on_dates(
                target, rotated, dates, start, end
            ),
        }
        folds.append(
            {
                "name": name,
                "train_label_dates_used_for_discovery_only": [0, start - 1],
                "validation_dates": [start, end],
                "strict_future_date_order": True,
                "all_rows_metric": metric,
                "complete_formula_rows_metric": exact_only,
                "complete_formula_fraction": float(complete[fold_mask].mean()),
                "placebos": placebos,
            }
        )

    fold_scores = [row["all_rows_metric"]["mean"] for row in folds]
    strongest_placebo = max(
        p["mean"] for row in folds for p in row["placebos"].values()
    )
    promotion = {
        "baseline_oof": BASELINE_OOF,
        "minimum_formula_fold": float(min(fold_scores)),
        "minimum_margin_over_baseline": float(min(fold_scores) - BASELINE_OOF),
        "strongest_placebo": float(strongest_placebo),
        "minimum_margin_over_strongest_placebo": float(min(fold_scores) - strongest_placebo),
        "gate": {
            "each_fold_above_0_95": bool(min(fold_scores) > 0.95),
            "each_fold_beats_baseline_by_0_80": bool(min(fold_scores) - BASELINE_OOF > 0.80),
            "beats_all_placebos_by_0_25": bool(min(fold_scores) - strongest_placebo > 0.25),
        },
    }
    promotion["passed"] = bool(all(promotion["gate"].values()))
    return {
        "formula": "sum(log(f_2[code,date+k])) for k in {2,3,4,5}; rank-equivalent to product",
        "uses_target_at_inference": False,
        "folds": folds,
        "promotion": promotion,
    }


def load_fallback_predictions(
    path: Path, test: pd.DataFrame
) -> tuple[np.ndarray, dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with zipfile.ZipFile(path) as archive:
        bad = archive.testzip()
        if bad is not None or archive.namelist() != ["submission.csv"]:
            raise ValueError(f"fallback ZIP invalid: bad={bad}, members={archive.namelist()}")
        with archive.open("submission.csv") as handle:
            fallback = pd.read_csv(handle, usecols=KEY_COLUMNS + [PREDICTION])
    if fallback.duplicated(KEY_COLUMNS).any():
        raise ValueError("fallback predictions contain duplicate keys")
    aligned = test[KEY_COLUMNS].merge(
        fallback, on=KEY_COLUMNS, how="left", validate="one_to_one"
    )
    values = aligned[PREDICTION].to_numpy(dtype=np.float32)
    if len(values) != len(test) or not np.isfinite(values).all():
        raise ValueError("fallback predictions fail row-count/finite audit")
    return values, {
        "path": str(path),
        "sha256": sha256_file(path),
        "rows": int(len(values)),
        "finite": True,
        "source_evidence": "v1 expanding-date OOF 0.089416; public 0.09057",
    }


def build_test_prediction(
    test: pd.DataFrame,
    fallback: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    dates = test["date"].to_numpy(dtype=np.int32)
    formula, complete = exact_date_log_product(test)
    score = np.where(complete, formula, 0.0).astype(np.float32)
    coverage_by_date = pd.Series(complete).groupby(test["date"], sort=False).mean()
    formula_dates = coverage_by_date[coverage_by_date > 0.0].index.to_numpy(dtype=np.int32)
    tail_dates = coverage_by_date[coverage_by_date == 0.0].index.to_numpy(dtype=np.int32)

    # For a date with at least one complete formula row, incomplete sparse rows
    # receive neutral formula score (validated above). For the terminal dates
    # where no row can see date+5, use the already-audited causal v1 candidate so
    # the official correlation never receives a constant vector.
    formula_rank = within_date_rank(score, dates)
    fallback_rank = within_date_rank(fallback, dates)
    prediction = formula_rank.copy()
    if len(tail_dates):
        tail_mask = np.isin(dates, tail_dates)
        prediction[tail_mask] = fallback_rank[tail_mask]
    prediction = within_date_rank(prediction, dates)
    return prediction, {
        "complete_row_fraction": float(complete.mean()),
        "formula_dates": int(len(formula_dates)),
        "formula_date_range": [int(formula_dates.min()), int(formula_dates.max())],
        "tail_fallback_dates": [int(v) for v in tail_dates],
        "tail_fallback_rows": int(np.isin(dates, tail_dates).sum()),
        "per_date_complete_fraction": {
            "min_on_formula_dates": float(coverage_by_date.loc[formula_dates].min()),
            "median_on_formula_dates": float(coverage_by_date.loc[formula_dates].median()),
            "max": float(coverage_by_date.max()),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--artifacts-dir", type=Path, default=Path("artifacts"))
    parser.add_argument(
        "--fallback-zip",
        type=Path,
        default=Path("artifacts/submissions/stocks_v2_model_blend_submission.zip"),
    )
    args = parser.parse_args()
    started = time.perf_counter()
    train_path = args.data_dir / "train_data.pkl"
    test_path = args.data_dir / "test_data.pkl"
    sample_path = args.data_dir / "sample_submission.csv"

    print("Loading train-only formula court...", flush=True)
    train_full = pd.read_pickle(train_path)
    train = train_full[["code", "date", "f_2", "y"]].copy()
    del train_full
    validation = validate_formula(train)
    print(json.dumps(validation["promotion"], indent=2), flush=True)
    if not validation["promotion"]["passed"]:
        raise RuntimeError("formula candidate failed promotion gate; refusing to build submission")
    del train
    gc.collect()

    print("Building static-batch test formula...", flush=True)
    test = pd.read_pickle(test_path)
    fallback, fallback_receipt = load_fallback_predictions(args.fallback_zip, test)
    prediction, prediction_receipt = build_test_prediction(test, fallback)
    del fallback
    gc.collect()

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
        args.artifacts_dir / "submissions" / "stocks_v2_exact_future_formula_submission.csv",
        args.artifacts_dir / "submissions" / "stocks_v2_exact_future_formula_submission.zip",
    )

    receipt = {
        "schema_version": "stocks-v2-formula-candidate-v1",
        "competition": COMPETITION,
        "created_at_utc": utc_now(),
        "external_submission_executed": False,
        "decision": "DIAGNOSTIC_ONLY_DO_NOT_SUBMIT",
        "validation": validation,
        "test_prediction": prediction_receipt,
        "fallback": fallback_receipt,
        "submission_audit": audit,
        "submission_artifact": artifact,
        "rules_boundary": {
            "official_description": "Use the training set to train your model and predict with the features data in the test set.",
            "official_specific_rule": "Don't cheat!",
            "sequential_serving_constraint_stated": False,
            "risk": "Formula consumes later test feature dates. They are present in the static test file, but this is non-causal in a live forecast interpretation.",
            "required_before_submission": "parent approval plus human/organizer determination that cross-date test-feature use is intended",
            "parent_decision": "do not submit the future-row formula; retain only as target-construction diagnostic",
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
        "runtime_seconds": float(time.perf_counter() - started),
    }
    receipt_path = args.artifacts_dir / "receipts" / "stocks_v2_formula_candidate.json"
    atomic_json(receipt_path, receipt)
    print(
        json.dumps(
            {
                "receipt": str(receipt_path),
                "artifact": artifact["zip_path"],
                "sha256": artifact["zip_sha256"],
                "external_submission_executed": False,
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
