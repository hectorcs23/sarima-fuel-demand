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
    python src/backtest_prices.py            # writes results/ and docs/img/backtest.png
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

    # 3. figure: error relative to the naive benchmark, and interval coverage
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    names = list(MODELS)
    colors = {"AR(2) levels (AIC pick)": "#d9534f", "ARIMA(p,1,q) (AIC pick)": "#2a6fdb",
              "Naive (random walk)": "#9aa4b1", "RW + drift": "#2e9e5b"}
    cols = list(df.columns)
    x = np.arange(len(cols)); w = 0.2

    for ax, h in zip(axes[:2], HORIZONS):
        sub = out[out.h == h]
        for i, name in enumerate(names):
            ratio = []
            for c in cols:
                mae = sub[(sub.series == c) & (sub.model == name)]["MAE"].iloc[0]
                base = sub[(sub.series == c) & (sub.model == "Naive (random walk)")]["MAE"].iloc[0]
                ratio.append(mae / base)
            ax.bar(x + (i - 1.5) * w, ratio, w, label=name, color=colors[name])
        ax.axhline(1.0, color="#333", ls="--", lw=1)
        ax.set_xticks(x); ax.set_xticklabels(cols)
        ax.set_ylim(0.8, 1.25)
        ax.set_ylabel("MAE / MAE of naive  (below 1 = better)")
        ax.set_title(f"{h}-week horizon", loc="left", fontweight="bold", fontsize=12)
    axes[0].legend(fontsize=8.5, loc="upper left")

    sub = out[out.h == 4]
    for i, name in enumerate(names):
        cov = [sub[(sub.series == c) & (sub.model == name)]["coverage95_%"].iloc[0] for c in cols]
        axes[2].bar(x + (i - 1.5) * w, cov, w, color=colors[name])
    axes[2].axhline(95, color="#333", ls="--", lw=1)
    axes[2].text(len(cols) - 0.5, 96, "nominal 95 %", fontsize=9)
    axes[2].set_xticks(x); axes[2].set_xticklabels(cols)
    axes[2].set_ylim(0, 108); axes[2].set_ylabel("% of weeks inside the interval")
    axes[2].set_title("Interval coverage (4 weeks)", loc="left", fontweight="bold", fontsize=12)
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=.25)
    fig.tight_layout()
    fig.savefig(FIGS / "backtest.png", dpi=130)
    print(f"\nwrote {RESULTS}/backtest_metrics.csv, {RESULTS}/adf_tests.csv, {FIGS}/backtest.png")


if __name__ == "__main__":
    main()
