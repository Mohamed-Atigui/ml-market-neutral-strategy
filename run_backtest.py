from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf

from src.strategy import (
    mean_rank_ic,
    momentum_baseline_backtest,
    performance_metrics,
    walk_forward_backtest,
)

UNIVERSE = ["XLB", "XLC", "XLE", "XLF", "XLI", "XLK", "XLP", "XLRE", "XLU", "XLV", "XLY"]
BENCHMARK = "SPY"
START = "2017-01-01"
OUT = Path("results")
OUT.mkdir(exist_ok=True)


def download_close(tickers):
    raw = yf.download(tickers, start=START, auto_adjust=True, progress=False, threads=True)
    close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]]
    if isinstance(close, pd.Series):
        close = close.to_frame()
    if close.shape[1] == 1 and len(tickers) == 1:
        close.columns = tickers
    return close.dropna(how="all").ffill().dropna()


prices = download_close(UNIVERSE)
spy = download_close([BENCHMARK])[BENCHMARK].reindex(prices.index).ffill().dropna()
prices = prices.reindex(spy.index).ffill().dropna()
prices.assign(SPY=spy).to_csv(OUT / "market_data.csv")

ml = walk_forward_backtest(
    prices,
    min_train_months=24,
    alpha=1.0,
    n_long=3,
    n_short=3,
    cost_bps=5.0,
)
strategy = ml["returns"]
ml["predictions"].to_csv(OUT / "predictions.csv", index=False)

baseline = momentum_baseline_backtest(prices, strategy.index, cost_bps=5.0)

# SPY next-month returns aligned to the same signal dates as the strategy.
spy_monthly = spy.loc[spy.groupby(spy.index.to_period("M")).tail(1).index]
spy_forward = spy_monthly.shift(-1) / spy_monthly - 1.0
spy_forward = spy_forward.reindex(strategy.index).dropna()
common = strategy.index.intersection(spy_forward.index)
strategy = strategy.loc[common]
baseline = baseline.loc[common]
spy_forward = spy_forward.loc[common]

metrics = pd.DataFrame(
    {
        "ridge_ml_net": performance_metrics(strategy["net_return"], strategy["turnover"]),
        "momentum_baseline_net": performance_metrics(baseline["net_return"], baseline["turnover"]),
        "SPY": performance_metrics(spy_forward),
    }
)
metrics.loc["mean_rank_ic", "ridge_ml_net"] = mean_rank_ic(ml["predictions"])
metrics.to_csv(OUT / "metrics.csv")

split = "2024-01-01"
period_metrics = pd.DataFrame(
    {
        "train_pre_2024": performance_metrics(strategy.loc[:"2023-12-31", "net_return"]),
        "test_2024_present": performance_metrics(strategy.loc[split:, "net_return"]),
    }
)
period_metrics.to_csv(OUT / "train_test_metrics.csv")

returns = pd.DataFrame(
    {
        "ridge_ml_net": strategy["net_return"],
        "momentum_baseline_net": baseline["net_return"],
        "SPY": spy_forward,
    }
)
returns.to_csv(OUT / "monthly_returns.csv")

equity = (1.0 + returns).cumprod()
equity.to_csv(OUT / "equity_curve.csv")

plt.figure(figsize=(10, 5))
equity.plot(ax=plt.gca())
plt.title("Walk-forward Ridge strategy vs baselines")
plt.ylabel("Growth of $1")
plt.tight_layout()
plt.savefig(OUT / "equity_curve.png", dpi=160)
plt.close()

strategy_equity = equity["ridge_ml_net"]
drawdown = strategy_equity / strategy_equity.cummax() - 1.0
plt.figure(figsize=(10, 4))
drawdown.plot(ax=plt.gca())
plt.title("Ridge ML strategy drawdown")
plt.ylabel("Drawdown")
plt.tight_layout()
plt.savefig(OUT / "drawdown.png", dpi=160)
plt.close()

print("Full-sample metrics")
print(metrics.round(4))
print("\nChronological split")
print(period_metrics.round(4))
