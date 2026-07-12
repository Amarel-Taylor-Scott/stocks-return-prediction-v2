from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pandas as pd


MODULE_PATH = Path(__file__).parents[1] / "src" / "run_pipeline.py"
SPEC = importlib.util.spec_from_file_location("stocks_v2_pipeline", MODULE_PATH)
pipeline = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = pipeline
SPEC.loader.exec_module(pipeline)


def test_daily_rank_ic_is_one_for_monotone_predictions() -> None:
    dates = np.repeat(np.arange(3), 5)
    y = np.tile(np.arange(5), 3).astype(float)
    result = pipeline.daily_rank_ic(y, 3.0 * y + 7.0, dates)
    assert result["dates"] == 3
    assert np.isclose(result["mean"], 1.0)


def test_expanding_prior_never_reads_current_or_future_target() -> None:
    code_ids = np.array([0, 1, 0, 1, 0], dtype=np.int32)
    target = np.array([0.3, -0.2, 0.1, 0.4, -0.5], dtype=np.float32)
    prior, coverage = pipeline.build_expanding_code_prior(target, code_ids, smoothing=0.0)
    assert np.isclose(prior[0], 0.0)
    assert np.isclose(prior[1], 0.0)
    assert np.isclose(prior[2], target[0])
    assert np.isclose(prior[3], target[1])
    assert np.isclose(prior[4], np.mean([target[0], target[2]]))
    assert coverage[4] > coverage[2]


def test_submission_audit_requires_exact_sample_order() -> None:
    sample = pd.DataFrame(
        {
            "id": [0, 1],
            "code": ["a", "b"],
            "date": [2, 2],
            "y_pred": [0.1, -0.1],
        }
    )
    test = sample.drop(columns=["id", "y_pred"]).assign(
        f_0=1.0,
        f_1=1.0,
        f_2=1.0,
        f_3=1,
        f_4=1.0,
        f_5=1.0,
        f_6=1.0,
    )
    result = pipeline.audit_submission(sample.copy(), sample, test)
    assert all(result["checks"].values())


def test_fold_specs_are_strictly_temporal() -> None:
    for fold in pipeline.FOLDS:
        assert fold.train_end < fold.valid_start <= fold.valid_end
