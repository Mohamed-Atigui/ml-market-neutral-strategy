from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf

from src.strategy import backtest, performance_metrics

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
spy_prices = download_close([BENCHMARK])[BENCHMARK].reindex(prices.index).ffill().dropna()
prices = prices.reindex(spy_prices.index).ffill().dropna()

bt = backtest(
    prices,
    lookback=126,
    skip=21,
    vol_window=63,
    target_vol=0.10,
    max_leverage=2.0,
    cost_bps=5.0,
    rebalance_frequency="monthly",
)

spy_returns = spy_prices.pct_change().fillna(0.0)

metrics = pd.DataFrame(
    {
        "strategy_gross": performance_metrics(bt["gross_returns"], bt["turnover"]),
        "strategy_net": performance_metrics(bt["net_returns"], bt["turnover"]),
        "SPY": performance_metrics(spy_returns),
    }
)
metrics.to_csv(OUT / "metrics.csv")

split = "2024-01-01"
period_metrics = pd.DataFrame(
    {
        "train_2017_2023": performance_metrics(bt["net_returns"].loc[:"2023-12-31"]),
        "test_2024_present": performance_metrics(bt["net_returns"].loc[split:]),
    }
)
period_metrics.to_csv(OUT / "train_test_metrics.csv")

eq = pd.DataFrame(
    {
        "strategy_gross": (1 + bt["gross_returns"]).cumprod(),
        "strategy_net": (1 + bt["net_returns"]).cumprod(),
        "SPY": (1 + spy_returns).cumprod(),
    }
)
eq.to_csv(OUT / "equity_curve.csv")

plt.figure(figsize=(10, 5))
eq.plot(ax=plt.gca())
plt.title("Market-neutral momentum vs SPY")
plt.ylabel("Growth of $1")
plt.tight_layout()
plt.savefig(OUT / "equity_curve.png", dpi=160)
plt.close()

net_eq = eq["strategy_net"]
drawdown = net_eq / net_eq.cummax() - 1.0
plt.figure(figsize=(10, 4))
drawdown.plot(ax=plt.gca())
plt.title("Net strategy drawdown")
plt.ylabel("Drawdown")
plt.tight_layout()
plt.savefig(OUT / "drawdown.png", dpi=160)
plt.close()

print("Full-sample metrics")
print(metrics.round(4))
print("\nChronological split")
print(period_metrics.round(4))
