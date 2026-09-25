"""
backtest_demand.py
==================

Same out-of-sample test as `backtest_prices.py`, but on the weekly fuel *demand*
series for Nuevo Leon (gasoline and diesel), which is the input the inventory
optimization actually consumes.

Models compared, rolling origin, expanding window, h = 1 and 4 weeks:
    - SARIMA(0,1,2)(0,0,1)52   the project's specification: MA(2) + seasonal MA(1)
    - SARIMA by AIC            small grid re-selected on the first training window
    - Naive (random walk)      forecast = last observed week
    - Seasonal naive           forecast = same week one year ago
    - Mean of last 4 weeks

Reported: MAE, RMSE, MAPE and the realized coverage of the nominal 95 % interval.

Usage:
    python src/backtest_demand.py
    python src/backtest_demand.py --plot-only  # redraw the figures from results/
"""
from __future__ import annotations

import itertools
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "fuel_demand_weekly_NL.csv"
RESULTS = ROOT / "results"
FIGS = ROOT / "docs" / "img"
HORIZONS = (1, 4)
TEST_WEEKS = 60
MIN_TRAIN = 78
SEASON = 52


def _sarima(train, h, order, seasonal):
    res = sm.tsa.SARIMAX(train.reset_index(drop=True), order=order,
                         seasonal_order=seasonal, trend="n").fit(disp=False)
    fc = res.get_forecast(h)
    ci = fc.conf_int(alpha=0.05).iloc[-1]
    return fc.predicted_mean.iloc[-1], ci.iloc[0], ci.iloc[1]


def f_project(train, h):
    """MA(2) + seasonal MA(1) — the structure fitted in the original project."""
    return _sarima(train, h, (0, 1, 2), (0, 0, 1, SEASON))


def f_aic(train, h, cache={}):
    """Small (p,d,q) grid re-selected by AIC, seasonal MA(1) kept."""
    key = train.name
    if key not in cache:
        best, best_order = np.inf, (0, 1, 1)
        for p, d, q in itertools.product(range(3), (0, 1), range(3)):
            try:
                aic = sm.tsa.SARIMAX(train.reset_index(drop=True), order=(p, d, q),
                                     seasonal_order=(0, 0, 1, SEASON),
                                     trend="n").fit(disp=False).aic
            except Exception:
                continue
            if np.isfinite(aic) and aic < best:
                best, best_order = aic, (p, d, q)
        cache[key] = best_order
        print(f"   [{key}] AIC picks SARIMA{best_order}(0,0,1){SEASON}")
    return _sarima(train, h, cache[key], (0, 0, 1, SEASON))


def f_naive(train, h):
    last = train.iloc[-1]
    sigma = train.diff().dropna().std()
    band = 1.96 * sigma * np.sqrt(h)
    return last, last - band, last + band


def f_snaive(train, h):
    """Same week last year; interval from the spread of year-over-year changes."""
    idx = len(train) - SEASON + h - 1
    point = train.iloc[idx] if idx >= 0 else train.iloc[-1]
    resid = (train - train.shift(SEASON)).dropna()
    sigma = resid.std() if len(resid) > 5 else train.diff().dropna().std()
    return point, point - 1.96 * sigma, point + 1.96 * sigma


def f_ma4(train, h):
    point = train.iloc[-4:].mean()
    sigma = train.diff().dropna().std()
    band = 1.96 * sigma * np.sqrt(h)
    return point, point - band, point + band


MODELS = {
    "SARIMA(0,1,2)(0,0,1)52 (project)": f_project,
    "SARIMA by AIC": f_aic,
    "Naive (random walk)": f_naive,
    "Seasonal naive (t-52)": f_snaive,
    "Mean of last 4 weeks": f_ma4,
}


def backtest(series: pd.Series) -> pd.DataFrame:
    rows = []
    for h in HORIZONS:
        preds = {n: [] for n in MODELS}
        inside = {n: [] for n in MODELS}
        actuals = []
        for t in range(len(series) - TEST_WEEKS, len(series) - h + 1):
            train = series.iloc[:t]
            if len(train) < MIN_TRAIN:
                continue
            actual = series.iloc[t + h - 1]
            actuals.append(actual)
            for name, fn in MODELS.items():
                point, lo, hi = fn(train, h)
                preds[name].append(point)
                inside[name].append(lo <= actual <= hi)
        actuals = np.asarray(actuals, dtype=float)
        for name in MODELS:
            err = np.asarray(preds[name], dtype=float) - actuals
            rows.append({"series": series.name, "h": h, "model": name,
                         "MAE": np.abs(err).mean(),
                         "RMSE": np.sqrt((err ** 2).mean()),
                         "MAPE_%": np.abs(err / actuals).mean() * 100,
                         "coverage95_%": np.mean(inside[name]) * 100,
                         "n": len(actuals)})
    return pd.DataFrame(rows)


def main():
    df = pd.read_csv(DATA, parse_dates=["date"]).set_index("date")
    RESULTS.mkdir(exist_ok=True)

    adf = []
    for col in df.columns:
        for label, s in (("levels", df[col]), ("first differences", df[col].diff().dropna())):
            stat, p, *_ = adfuller(s, autolag="AIC")
            adf.append({"series": col, "on": label, "ADF": stat, "p_value": p,
                        "stationary_5%": p < 0.05})
    adf = pd.DataFrame(adf)
    adf.to_csv(RESULTS / "adf_tests_demand.csv", index=False)
    print(adf.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print()

    out = pd.concat([backtest(df[c].rename(c)) for c in df.columns], ignore_index=True)
    out.to_csv(RESULTS / "backtest_metrics_demand.csv", index=False)
    print()
    print(out.pivot_table(index=["series", "model"], columns="h",
                          values=["MAE", "MAPE_%", "coverage95_%"]).round(2).to_string())

    make_figures(out)
    print(f"\nwrote {RESULTS}/backtest_metrics_demand.csv and the backtest_demand_*.png figures in {FIGS}/")


# -------------------------------------------------------------------- figures
SHORT = {"SARIMA(0,1,2)(0,0,1)52 (project)": "SARIMA (project)", "SARIMA by AIC": "SARIMA (AIC)",
         "Naive (random walk)": "Naive", "Seasonal naive (t-52)": "Seasonal naive",
         "Mean of last 4 weeks": "Mean 4 wk"}


def make_figures(out: pd.DataFrame) -> None:
    """One figure per question: relative MAE at h = 1, at h = 4, and interval coverage."""
    FIGS.mkdir(parents=True, exist_ok=True)
    cols = list(dict.fromkeys(out["series"]))
    names = list(dict.fromkeys(out["model"]))
    x = np.arange(len(cols))
    w = 0.8 / len(names)

    for h in HORIZONS:
        sub = out[out.h == h]
        plt.figure()
        for i, name in enumerate(names):
            ratio = [sub[(sub.series == c) & (sub.model == name)]["MAE"].iloc[0] /
                     sub[(sub.series == c) & (sub.model == "Naive (random walk)")]["MAE"].iloc[0]
                     for c in cols]
            plt.plot(x + (i - (len(names) - 1) / 2) * w, ratio, "o", label=SHORT.get(name, name))
        plt.plot([-0.5, len(cols) - 0.5], [1, 1], "k--", label="Naive = 1")
        plt.xticks(x, cols)
        plt.xlim(-0.5, len(cols) + 0.9)  # empty column on the right for the legend
        plt.xlabel("Series")
        plt.ylabel("MAE / MAE of naive (below 1 = better)")
        plt.title(f"Fuel demand: out-of-sample MAE, {h}-week horizon")
        plt.legend(loc="center right")
        plt.savefig(FIGS / f"backtest_demand_h{h}.png")

    sub = out[out.h == max(HORIZONS)]
    plt.figure()
    for i, name in enumerate(names):
        cov = [sub[(sub.series == c) & (sub.model == name)]["coverage95_%"].iloc[0] for c in cols]
        plt.plot(x + (i - (len(names) - 1) / 2) * w, cov, "o", label=SHORT.get(name, name))
    plt.plot([-0.5, len(cols) - 0.5], [95, 95], "k--", label="Nominal 95 %")
    plt.xticks(x, cols)
    plt.xlim(-0.5, len(cols) + 0.9)
    plt.xlabel("Series")
    plt.ylabel("% of weeks inside the 95 % interval")
    plt.title(f"Fuel demand: interval coverage, {max(HORIZONS)}-week horizon")
    plt.legend(loc="center right")
    plt.savefig(FIGS / f"backtest_demand_coverage.png")


if __name__ == "__main__":
    import sys
    if "--plot-only" in sys.argv:      # redraw from the saved metrics, no refit
        make_figures(pd.read_csv(RESULTS / "backtest_metrics_demand.csv"))
    else:
        main()
