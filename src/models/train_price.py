"""Train the house-price model (LightGBM) with an honest, time-based evaluation.

Target: log(price / market_index) — i.e. how expensive a property is relative
to the national market *at the time of sale*. This lets a tree model handle
market-wide inflation, which trees cannot extrapolate on their own.

Evaluation: train on everything before the last 12 months, test on the last
12 months (simulates valuing new sales the model has never seen). Compared
against two baselines so the improvement is measurable.
"""
import json

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, median_absolute_error, r2_score

from src.config import PRICE_METRICS_PATH, PRICE_MODEL_PATH
from src.data.database import query
from src.models.features import (CATEGORICAL, FEATURES, add_sale_history, build_matrix, from_log_relative,
                                 market_index, to_log_relative)

PARAMS = dict(
    # L1 loss optimises the typical (median) error and is robust to outlier sales
    objective="regression_l1", learning_rate=0.05, num_leaves=63, min_child_samples=40,
    feature_fraction=0.9, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
    cat_smooth=20, verbose=-1, seed=42,
)


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "MAE_eur": float(mean_absolute_error(y_true, y_pred)),
        "MedianAE_eur": float(median_absolute_error(y_true, y_pred)),
        "MAPE_pct": float(np.mean(np.abs(y_pred - y_true) / y_true) * 100),
        "MdAPE_pct": float(np.median(np.abs(y_pred - y_true) / y_true) * 100),
        "within_10pct": float(np.mean(np.abs(y_pred - y_true) / y_true <= 0.10) * 100),
        "R2_log": float(r2_score(np.log(y_true), np.log(y_pred))),
    }


def train(test_months: int = 12) -> dict:
    df = query("SELECT * FROM sales ORDER BY sale_date")
    idx = market_index(df)
    df = df.join(idx, on="period")
    df = add_sale_history(df, idx)

    cutoff = df["period"].max() - pd.DateOffset(months=test_months - 1)
    train_df, test_df = df[df["period"] < cutoff], df[df["period"] >= cutoff]
    print(f"[train] train: {len(train_df):,} rows (< {cutoff:%Y-%m}), test: {len(test_df):,} rows")

    X_tr, cats = build_matrix(train_df)
    X_te, _ = build_matrix(test_df, cats)
    y_tr = to_log_relative(train_df["price_eur"], train_df["market_index"])
    y_te = to_log_relative(test_df["price_eur"], test_df["market_index"])

    # Validation = last 12 months of the training window, for early stopping only
    val_mask = (train_df["period"] >= cutoff - pd.DateOffset(months=12)).values
    # Recent years matter more for today's market: weight halves roughly every 4 years
    w = np.exp((train_df["year"] - train_df["year"].max()) / 6.0).values
    dtrain = lgb.Dataset(X_tr[~val_mask], y_tr[~val_mask], weight=w[~val_mask], categorical_feature=CATEGORICAL)
    dval = lgb.Dataset(X_tr[val_mask], y_tr[val_mask], categorical_feature=CATEGORICAL, reference=dtrain)
    booster = lgb.train(PARAMS, dtrain, num_boost_round=4000, valid_sets=[dval],
                        callbacks=[lgb.early_stopping(100, verbose=False)])
    best_iter = booster.best_iteration
    print(f"[train] early-stopped at {best_iter} trees")

    pred_te = from_log_relative(booster.predict(X_te, num_iteration=best_iter), test_df["market_index"])
    y_true = test_df["price_eur"].values

    # Baseline 1: county median price over the last 12 months of training data
    recent = train_df[train_df["period"] >= cutoff - pd.DateOffset(months=12)]
    county_med = recent.groupby("county")["price_eur"].median()
    base1 = test_df["county"].map(county_med).fillna(recent["price_eur"].median()).values
    # Baseline 2: county relative level x current market index (same trick as the model, no ML)
    rel = (train_df["price_eur"] / train_df["market_index"]).groupby(train_df["county"]).median()
    base2 = (test_df["county"].map(rel).fillna(1.0) * test_df["market_index"]).values

    results = {
        "test_period": f"{cutoff:%Y-%m} to {df['period'].max():%Y-%m}",
        "n_train": int(len(train_df)), "n_test": int(len(test_df)),
        "lightgbm": metrics(y_true, pred_te),
        "baseline_county_median": metrics(y_true, base1),
        "baseline_county_x_index": metrics(y_true, base2),
    }
    test_df = test_df.assign(pred=pred_te, abs_err=np.abs(pred_te - y_true))
    rep = test_df["prev_rel_price"].notna().values
    results["repeat_sales"] = {"share_of_test_pct": float(rep.mean() * 100),
                               "lightgbm": metrics(y_true[rep], pred_te[rep]),
                               "baseline_county_median": metrics(y_true[rep], base1[rep]),
                               "lightgbm_first_time_sales": metrics(y_true[~rep], pred_te[~rep])}
    results["mae_by_county"] = test_df.groupby("county")["abs_err"].mean().round(0).sort_values().to_dict()

    # Global feature importance = mean |SHAP| (LightGBM computes exact TreeSHAP natively)
    sample = X_te.sample(min(5000, len(X_te)), random_state=0)
    contrib = booster.predict(sample, num_iteration=best_iter, pred_contrib=True)[:, :-1]
    results["shap_importance"] = dict(sorted(
        zip(FEATURES, np.abs(contrib).mean(axis=0).round(4).tolist()), key=lambda kv: -kv[1]))

    # Refit on ALL data with the chosen number of trees for the deployed model
    X_all, cats = build_matrix(df)
    y_all = to_log_relative(df["price_eur"], df["market_index"])
    w_all = np.exp((df["year"] - df["year"].max()) / 6.0).values
    final = lgb.train(PARAMS, lgb.Dataset(X_all, y_all, weight=w_all, categorical_feature=CATEGORICAL),
                      num_boost_round=best_iter)

    latest_index = float(market_index(df).iloc[-1])
    joblib.dump({"booster": final, "categories": cats, "latest_index": latest_index,
                 "index_series": idx, "trained_until": f"{df['period'].max():%Y-%m}"}, PRICE_MODEL_PATH)
    PRICE_METRICS_PATH.write_text(json.dumps(results, indent=2))
    _log_mlflow(results)

    lg, b = results["lightgbm"], results["baseline_county_x_index"]
    print(f"[train] LightGBM  MAE €{lg['MAE_eur']:,.0f} | MAPE {lg['MAPE_pct']:.1f}% | within 10%: {lg['within_10pct']:.1f}%")
    print(f"[train] Baseline  MAE €{b['MAE_eur']:,.0f} | MAPE {b['MAPE_pct']:.1f}%")
    return results


def _log_mlflow(results: dict) -> None:
    """Optional experiment tracking: runs only if `pip install mlflow` was done."""
    try:
        import mlflow
    except ImportError:
        return
    mlflow.set_experiment("irish-house-prices")
    with mlflow.start_run():
        mlflow.log_params(PARAMS)
        mlflow.log_metrics({f"test_{k}": v for k, v in results["lightgbm"].items()})
        mlflow.log_artifact(str(PRICE_METRICS_PATH))


if __name__ == "__main__":
    train()
