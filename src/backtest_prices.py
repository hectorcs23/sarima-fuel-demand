"""
backtest_prices.py
==================

Out-of-sample check for the weekly fuel-price models.

What it answers, which AIC alone cannot:
  1. Is the series a random walk?  (ADF on levels and on first differences)
  2. Does the fitted model beat trivial baselines out of sample?
     Rolling origin, expanding window, horizons h = 1, 4 weeks:
        - AR(2) in levels            (the model selected by AIC in the project)
        - ARIMA(p,1,q) by AIC        (same search, but on the differenced series)
        - Naive / random walk        (forecast = last observed price)
        - Random walk with drift
  3. Do the 95 % intervals actually cover 95 % of the realized values?

Usage:
    python src/backtest_prices.py              # writes results/ and docs/img/backtest_*.png
    python src/backtest_prices.py --plot-only  # redraw the figures from results/
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
DATA = ROOT / "data" / "fuel_prices_weekly_MXN.csv"
RESULTS = ROOT / "results"
FIGS = ROOT / "docs" / "img"
HORIZONS = (1, 4)
TEST_WEEKS = 78          # last ~1.5 years used as rolling out-of-sample origins
MIN_TRAIN = 104          # never fit on fewer than 2 years


# ---------------------------------------------------------------- forecasters
def f_ar2(train: pd.Series, h: int):
    """AR(2) in levels with constant — the specification chosen by AIC in the project."""
    res = sm.tsa.SARIMAX(train, order=(2, 0, 0), trend="c").fit(disp=False)
    fc = res.get_forecast(h)
    ci = fc.conf_int(alpha=0.05).iloc[-1]
    return fc.predicted_mean.iloc[-1], ci.iloc[0], ci.iloc[1]


def f_arima_d1(train: pd.Series, h: int, cache={}):
    """ARIMA(p,1,q), orders picked once by AIC on the first training window."""
    key = train.name
    if key not in cache:
        best, order = np.inf, (0, 1, 0)
        for p, q in itertools.product(range(3), range(3)):
            try:
                aic = sm.tsa.SARIMAX(train, order=(p, 1, q), trend="n").fit(disp=False).aic
            except Exception:
                continue
            if aic < best:
                best, order = aic, (p, 1, q)
        cache[key] = order
    res = sm.tsa.SARIMAX(train, order=cache[key], trend="n").fit(disp=False)
    fc = res.get_forecast(h)
    ci = fc.conf_int(alpha=0.05).iloc[-1]
    return fc.predicted_mean.iloc[-1], ci.iloc[0], ci.iloc[1]


def f_naive(train: pd.Series, h: int):
    """Random walk: the last observed price, with the interval implied by sigma*sqrt(h)."""
    last = train.iloc[-1]
    sigma = train.diff().dropna().std()
    band = 1.96 * sigma * np.sqrt(h)
    return last, last - band, last + band


def f_drift(train: pd.Series, h: int):
    """Random walk with drift estimated over the whole training window."""
    last, n = train.iloc[-1], len(train)
    drift = (train.iloc[-1] - train.iloc[0]) / (n - 1)
    sigma = (train.diff().dropna() - drift).std()
    band = 1.96 * sigma * np.sqrt(h)
    point = last + drift * h
    return point, point - band, point + band


MODELS = {
    "AR(2) levels (AIC pick)": f_ar2,
    "ARIMA(p,1,q) (AIC pick)": f_arima_d1,
    "Naive (random walk)": f_naive,
    "RW + drift": f_drift,
}


# --------------------------------------------------------------------- driver
def backtest(series: pd.Series) -> pd.DataFrame:
    rows = []
    origins = range(len(series) - TEST_WEEKS, len(series) - max(HORIZONS) + 1)
    for h in HORIZONS:
        preds = {name: [] for name in MODELS}
        actuals, inside = [], {name: [] for name in MODELS}
        for t in origins:
            train = series.iloc[:t]
            if len(train) < MIN_TRAIN:
                continue
            actual = series.iloc[t + h - 1]
            actuals.append(actual)
            for name, fn in MODELS.items():
                point, lo, hi = fn(train, h)
                preds[name].append(point)
                inside[name].append(lo <= actual <= hi)
        actuals = np.asarray(actuals)
        for name in MODELS:
            err = np.asarray(preds[name]) - actuals
            rows.append({
                "series": series.name, "h": h, "model": name,
                "MAE": np.abs(err).mean(),
                "RMSE": np.sqrt((err ** 2).mean()),
                "MAPE_%": (np.abs(err / actuals)).mean() * 100,
                "coverage95_%": np.mean(inside[name]) * 100,
                "n": len(actuals),
            })
    return pd.DataFrame(rows)


def main():
    df = pd.read_csv(DATA, parse_dates=["date"]).set_index("date")
    RESULTS.mkdir(exist_ok=True)
    FIGS.mkdir(parents=True, exist_ok=True)

    # 1. stationarity
    adf = []
    for col in df.columns:
        for label, s in (("levels", df[col]), ("first differences", df[col].diff().dropna())):
            stat, p, *_ = adfuller(s, autolag="AIC")
            adf.append({"series": col, "on": label, "ADF": stat, "p_value": p,
                        "stationary_5%": p < 0.05})
    adf = pd.DataFrame(adf)
    adf.to_csv(RESULTS / "adf_tests.csv", index=False)
    print(adf.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    # 2. rolling-origin backtest
    out = pd.concat([backtest(df[c].rename(c)) for c in df.columns], ignore_index=True)
    out.to_csv(RESULTS / "backtest_metrics.csv", index=False)
    print()
    print(out.pivot_table(index=["series", "model"], columns="h",
                          values=["MAE", "coverage95_%"]).round(3).to_string())

    make_figures(out)
    print(f"\nwrote {RESULTS}/backtest_metrics.csv and the backtest_*.png figures in {FIGS}/")


# -------------------------------------------------------------------- figures
SHORT = {"AR(2) levels (AIC pick)": "AR(2)", "ARIMA(p,1,q) (AIC pick)": "ARIMA(p,1,q)",
         "Naive (random walk)": "Naive", "RW + drift": "RW + drift"}


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
        plt.title(f"Fuel prices: out-of-sample MAE, {h}-week horizon")
        plt.legend(loc="center right")
        plt.savefig(FIGS / f"backtest_h{h}.png")

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
    plt.title(f"Fuel prices: interval coverage, {max(HORIZONS)}-week horizon")
    plt.legend(loc="center right")
    plt.savefig(FIGS / "backtest_coverage.png")


if __name__ == "__main__":
    import sys
    if "--plot-only" in sys.argv:      # redraw from the saved metrics, no refit
        make_figures(pd.read_csv(RESULTS / "backtest_metrics.csv"))
    else:
        main()
