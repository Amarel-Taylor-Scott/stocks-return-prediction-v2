from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


SRC = Path(__file__).parents[1] / "src"
sys.path.insert(0, str(SRC))
import run_formula_candidate as formula  # noqa: E402
import run_causal_distilled as distilled  # noqa: E402
import run_pipeline as pipeline  # noqa: E402


def test_exact_date_join_does_not_skip_over_missing_date() -> None:
    rows = []
    for code, dates in (("a", range(7)), ("b", (0, 1, 2, 4, 5, 6))):
        for date in dates:
            rows.append({"code": code, "date": date, "f_2": float(np.exp(date / 100.0))})
    frame = pd.DataFrame(rows).sort_values(["date", "code"], kind="stable").reset_index(drop=True)
    score, complete = formula.exact_date_log_product(frame)
    a0 = frame.index[(frame.code == "a") & (frame.date == 0)][0]
    b0 = frame.index[(frame.code == "b") & (frame.date == 0)][0]
    assert complete[a0]
    assert np.isclose(score[a0], (2 + 3 + 4 + 5) / 100.0)
    assert not complete[b0]  # exact date+3 is absent; next-observation is forbidden


def test_future_formula_recovers_constructed_daily_ranks() -> None:
    rng = np.random.default_rng(20260712)
    rows = []
    for date in range(12):
        for code in range(30):
            rows.append(
                {
                    "code": f"s_{code}",
                    "date": date,
                    "f_2": float(np.exp(rng.normal(0.0, 0.02))),
                }
            )
    frame = pd.DataFrame(rows)
    score, complete = formula.exact_date_log_product(frame)
    dates = frame.date.to_numpy()
    mask = complete
    target = 7.0 * score + dates * 0.3  # date-specific offsets cannot change daily ranks
    metric = pipeline.daily_rank_ic(target[mask], score[mask], dates[mask])
    assert np.isclose(metric["mean"], 1.0)


def test_within_date_rotation_is_distribution_matched_but_misaligned() -> None:
    dates = np.repeat(np.arange(2), 5)
    values = np.tile(np.arange(5, dtype=np.float32), 2)
    rotated = formula.rotate_within_date(values, dates)
    for date in range(2):
        mask = dates == date
        np.testing.assert_array_equal(np.sort(rotated[mask]), np.sort(values[mask]))
        assert not np.array_equal(rotated[mask], values[mask])


def test_terminal_dates_use_nonconstant_fallback() -> None:
    rows = []
    fallback = []
    for date in range(10):
        for code in range(6):
            rows.append(
                {
                    "code": f"s_{code}",
                    "date": date,
                    "f_2": 1.0 + 0.001 * (date + code + 1),
                }
            )
            fallback.append(float(code))
    test = pd.DataFrame(rows)
    prediction, receipt = formula.build_test_prediction(
        test, np.asarray(fallback, dtype=np.float32)
    )
    assert receipt["tail_fallback_dates"] == [5, 6, 7, 8, 9]
    for date in receipt["tail_fallback_dates"]:
        values = prediction[test.date.to_numpy() == date]
        assert np.unique(values).size == 6


def test_distilled_folds_purge_full_label_horizon() -> None:
    for fold in distilled.FOLDS:
        distilled.validate_fold_contract(fold)
        assert fold.purged_label_end + distilled.HORIZON_END <= fold.nominal_train_end
        assert fold.nominal_train_end < fold.valid_start
