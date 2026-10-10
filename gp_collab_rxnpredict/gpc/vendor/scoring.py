"""Vendored from GP_collab `code_python/training/scoring_validation.py`.

Trimmed to the pieces this package uses: the per-fold worker, the non-grouped
cross-validation branch, and the seed-level score aggregation. The grouped
(LeaveOneGroupOut / matched-IID-resample) branch was dropped because every
evaluation method here arrives as an explicit list of (train, test) index pairs
-- including LOLO -- so all three run through the one non-grouped path and
produce directly comparable pooled out-of-fold predictions.
"""
from typing import Optional, Union, Dict
from time import perf_counter
from joblib import Parallel, delayed
import numpy as np
import pandas as pd
from collections import defaultdict
import copy

from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score

from .uq import (
    compute_ece,
    compute_cdf_ama,
    compute_cvpp_ama,
    gaussian_nll,
    compute_RUSC,
    compute_Cv,
    compute_sharpness
    )


def split_for_training(data, indices):
    """From GP_collab `code_python/training/utils.py`."""
    indices = np.asarray(indices)
    if isinstance(data, (pd.DataFrame, pd.Series)):
        return data.iloc[indices].copy()
    if isinstance(data, np.ndarray):
        return data[indices]
    raise ValueError(f"Unsupported data type {type(data)}.")


def _fit_predict_score(
    estimator,
    model_type,
    X,
    y,
    train_idx,
    test_idx,
    scoring,
    return_ls: bool,
    UQ: bool,
    return_estimator: bool = False,
    return_tree_importances: bool = False,
):
    """
    Shared GP/tree worker that runs inside a parallel worker:
    - clone estimator
    - fit on train
    - predict on test
    - compute scores with provided scorers
    """
    fold_start = perf_counter()

    if "mgk" in model_type.lower():
        est = copy.deepcopy(estimator)
    else:
        est = clone(estimator)

    # Use safe row selector for all data types
    X_train = split_for_training(X, train_idx)
    X_test  = split_for_training(X, test_idx)
    y_train = split_for_training(y, train_idx)
    y_test  = split_for_training(y, test_idx)

    est.fit(X_train, y_train)

    # All project GP/tree wrappers return the same prediction dictionary.
    y_result = est.predict(X_test, return_std=UQ)
    if not isinstance(y_result, dict) or "y_pred" not in y_result:
        raise TypeError(
            f"{type(est).__name__}.predict must return a dictionary containing "
            "'y_pred' and 'y_std'."
        )

    y_result["y_pred"] = np.asarray(y_result["y_pred"]).ravel()
    if y_result.get("y_std") is not None:
        y_result["y_std"] = np.asarray(y_result["y_std"]).ravel()

    results = {}
    fitted_regressor = (
        est.named_steps["regressor"]
        if isinstance(est, Pipeline)
        else est
    )

    if return_ls:
        results["lengthscale"] = fitted_regressor._get_lengthscale()

    if return_tree_importances:
        results["feature_importance_MDI"] = fitted_regressor._get_MDI()
        results["feature_importance_SHAP"] = fitted_regressor._get_SHAP()

    if return_estimator:
        results["estimator"] = est

    y_test = np.asarray(y_test).ravel()
    for name, scorer in scoring.items():
        results[name] = scorer(y_test, y_result["y_pred"])

    if UQ and y_result.get("y_std") is not None:
        UQ_scorers = {
            "ece": compute_ece,
            "RUSC": compute_RUSC,
            "cdf_ama": compute_cdf_ama,
            "cvpp_ama": compute_cvpp_ama,
            "nll": gaussian_nll,
            "Cv": compute_Cv,
            "sharpness": compute_sharpness
        }
        for name, uq_scorer in UQ_scorers.items():
            if name in {"Cv", "sharpness"}:
                results[name] = float(uq_scorer(y_result["y_std"]))
            else:
                results[name] = float(uq_scorer(y_test, y_result["y_pred"], y_result["y_std"]))

    results["run_time_sec"] = perf_counter() - fold_start

    return test_idx, y_result, results
def cross_validate(
    estimator,
    model_type,
    X,
    y,
    cv,
    scoring,
    UQ,
    n_jobs,
    return_ls,
    return_estimator,
    return_tree_importances,
):
    """Fold-parallel CV over the splits `cv` yields, pooling out-of-fold predictions."""
    n_samples = len(y)

    parallel_kwargs = {
        "n_jobs": n_jobs,
        "verbose": 0,
    }

    if "gp" in model_type.lower() or "mgk" in model_type.lower():
        parallel_kwargs["require"] = "sharedmem"
    else:
        parallel_kwargs["pre_dispatch"] = "all"

    splits = cv.split(X, y)

    parallel_results = Parallel(**parallel_kwargs)(
        delayed(_fit_predict_score)(
            estimator,
            model_type,
            X,
            y,
            train_idx,
            test_idx,
            scoring,
            return_ls,
            UQ,
            return_estimator,
            return_tree_importances,
        )
        for train_idx, test_idx in splits
    )

    scores = defaultdict(list)

    predictions = {
        "y_pred": np.full(n_samples, np.nan),
        "y_std": np.full(n_samples, np.nan),
    }

    for test_idx, y_result, fold_scores in parallel_results:

        predictions["y_pred"][test_idx] = y_result["y_pred"]

        if y_result.get("y_std") is not None:
            predictions["y_std"][test_idx] = y_result["y_std"]

        for key, val in fold_scores.items():

            score_key = (
                key
                if key == "estimator"
                else f"test_{key}"
            )

            scores[score_key].append(val)


    return scores, predictions


def process_scores(
    scores: dict[int, dict[str, float]],
) -> dict[Union[int, str], dict[str, float]]:
    
    first_key = list(scores.keys())[0]
    
    score_types: list[str] = [
        key for key in scores[first_key].keys() 
        if (
            key.startswith("test_")
            and "lengthscale" not in key
            and "feature_importance" not in key
        )
    ]

    arr = np.array(scores[first_key]["test_r2"])
    
    if arr.ndim > 1 and arr.shape[1] > 1:
        avg_r2 = np.round(np.mean(np.vstack([arr for seed in scores.values() for arr in seed["test_r2"]]), axis=0), 3)
        stdev_r2 = np.round(np.std(np.vstack([arr for seed in scores.values() for arr in seed["test_r2"]]), axis=0), 3)
        print("Average scores:\t", f"r2: {avg_r2}±{stdev_r2}")
        
        avgs: list[float] = [
            np.mean(np.vstack([arr for seed in scores.values() for arr in seed[score]]), axis=0) for score in score_types
        ]
        stdevs: list[float] = [
            np.std(np.vstack([arr for seed in scores.values() for arr in seed[score]]), axis=0) for score in score_types
        ]
    else:
        avg_rmse = round(np.mean([seed["test_rmse"] for seed in scores.values()]), 2)
        stdev_rmse = round(np.std([seed["test_rmse"] for seed in scores.values()]), 2)
        avg_r2 = round(np.mean([seed["test_r2"] for seed in scores.values()]), 2)
        stdev_r2 = round(np.std([seed["test_r2"] for seed in scores.values()]), 2)
        print("Average scores:\t",
            f"rmse: {abs(avg_rmse)}±{stdev_rmse}\t",
            f"r2: {avg_r2}±{stdev_r2}")

        avgs: list[float] = [
            np.mean([seed[score] for seed in scores.values()]) for score in score_types
        ]
        stdevs: list[float] = [
            np.std([seed[score] for seed in scores.values()]) for score in score_types
        ]

    clean_score_types: list[str] = [score.replace("test_", "") for score in score_types]
    for score, avg, stdev in zip(clean_score_types, avgs, stdevs):
        scores[f"{score}_avg"] = abs(avg) if score in ["rmse", "mae"] else avg
        scores[f"{score}_stdev"] = stdev
    
    if arr.ndim > 1 and arr.shape[1] > 1:
        for score in clean_score_types:
            scores[f"{score}_avg_aggregate"] = np.mean(scores[f"{score}_avg"])
            scores[f"{score}_stdev_aggregate"] = np.mean(scores[f"{score}_stdev"])

    return scores


def _average_ls(ls_data: Dict) -> None:
    feature_values = defaultdict(list)
    for _, folds in ls_data.items():
        for _, features in folds.items():
            for feature, value in features.items():
                feature_values[feature].append(value)

    stats = {
        feature: {
            "mean": float(np.mean(values)),
            "std": float(np.std(values, ddof=1))
        }
        for feature, values in feature_values.items()
    }

    ls_data["aggregated_ls"] = stats

    return ls_data


def cross_validate_regressor(
    regressor,
    model_type: str,
    X, y,
    cv,
    UQ: bool = False,
    return_ls: bool = False,
    return_estimator: bool = False,
    return_tree_importances: bool = False,
    n_jobs: int = 1,
    ) -> tuple[dict[str, float], dict[str, np.ndarray]]:

        scorers = {
            "rmse": root_mean_squared_error,
            "mae": mean_absolute_error,
            "r2": r2_score,
            }

        return cross_validate(
            regressor,
            model_type,
            X,
            y,
            cv=cv,
            scoring=scorers,
            n_jobs=n_jobs,
            return_ls=return_ls,
            UQ=UQ,
            return_estimator=return_estimator,
            return_tree_importances=return_tree_importances,
            )
