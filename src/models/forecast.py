"""County-level price forecasting with a proper backtest.

For each county:
  1. Build a monthly median-price series (3-month rolling median to cut noise).
  2. Backtest 3 models on the last 12 months: naive, seasonal-naive-with-drift, damped Holt (ETS).
  3. Pick the model with the lowest backtest MAPE, refit on all data and forecast 12 months
     with an 80% interval derived from in-sample residuals.
"""
import json
import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from src.config import FORECAST_METRICS_PATH, FORECAST_PATH
from src.data.database import query

HORIZON = 12
Z80 = 1.2816


def county_series() -> dict[str, pd.Series]:
    df = query("SELECT county, period, median_price, n_sales FROM monthly_county ORDER BY period")
    out = {}
    for county, g in df.groupby("county"):
        s = g.set_index("period")["median_price"].asfreq("MS").interpolate()
        out[county] = s.rolling(3, min_periods=1).median()
    # National series too
    nat = query("SELECT period, MEDIAN(price_eur) AS p FROM sales GROUP BY period ORDER BY period")
    out["Ireland (all)"] = nat.set_index("period")["p"].asfreq("MS").interpolate().rolling(3, min_periods=1).median()
    return out


def fc_naive(s: pd.Series, h: int) -> np.ndarray:
    return np.repeat(s.iloc[-1], h)


def fc_seasonal_drift(s: pd.Series, h: int) -> np.ndarray:
    """Last year's value for the same month, scaled by the year-on-year growth rate."""
    growth = s.iloc[-1] / s.iloc[-13]
    return np.array([s.iloc[-12 + (i % 12)] * growth ** (1 + i // 12) for i in range(h)])


def fit_ets(s: pd.Series):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return ExponentialSmoothing(np.log(s.values), trend="add", damped_trend=True).fit()


def fc_ets(s: pd.Series, h: int) -> np.ndarray:
    return np.exp(fit_ets(s).forecast(h))


MODELS = {"naive": fc_naive, "seasonal_drift": fc_seasonal_drift, "damped_holt": fc_ets}


def mape(a, f) -> float:
    return float(np.mean(np.abs(np.asarray(f) - np.asarray(a)) / np.asarray(a)) * 100)


def run() -> pd.DataFrame:
    frames, report = [], {}
    for county, s in county_series().items():
        s = s.dropna()
        if len(s) < 36:
            continue
        train, test = s.iloc[:-HORIZON], s.iloc[-HORIZON:]
        scores = {name: mape(test.values, fn(train, HORIZON)) for name, fn in MODELS.items()}
        best = min(scores, key=scores.get)

        future_idx = pd.date_range(s.index[-1] + pd.offsets.MonthBegin(1), periods=HORIZON, freq="MS")
        point = MODELS[best](s, HORIZON)
        # Interval: spread of backtest % errors of the chosen model, widening with sqrt(h)
        bt = MODELS[best](train, HORIZON)
        sigma = max(float(np.sqrt(np.mean((np.log(test.values) - np.log(bt)) ** 2))), 0.02)
        width = Z80 * sigma * np.sqrt(np.arange(1, HORIZON + 1) / (HORIZON / 2))

        frames.append(pd.DataFrame({"county": county, "period": s.index, "value": s.values,
                                    "lower": np.nan, "upper": np.nan, "kind": "history"}))
        frames.append(pd.DataFrame({"county": county, "period": future_idx, "value": point,
                                    "lower": point * np.exp(-width), "upper": point * np.exp(width),
                                    "kind": "forecast"}))
        report[county] = {"backtest_mape_pct": {k: round(v, 2) for k, v in scores.items()}, "chosen": best,
                          "forecast_change_12m_pct": round((point[-1] / s.iloc[-1] - 1) * 100, 1)}

    out = pd.concat(frames, ignore_index=True)
    out.to_csv(FORECAST_PATH, index=False)
    FORECAST_METRICS_PATH.write_text(json.dumps(report, indent=2))
    avg = {m: np.mean([r["backtest_mape_pct"][m] for r in report.values()]) for m in MODELS}
    print("[forecast] mean backtest MAPE across counties: " + ", ".join(f"{k} {v:.1f}%" for k, v in avg.items()))
    return out


if __name__ == "__main__":
    run()
