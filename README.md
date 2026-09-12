# Cross-Sectional ML Market-Neutral Strategy

[![Quant research CI](https://github.com/Mohamed-Atigui/ml-market-neutral-strategy/actions/workflows/ci.yml/badge.svg)](https://github.com/Mohamed-Atigui/ml-market-neutral-strategy/actions/workflows/ci.yml)

A reproducible quantitative-research project that predicts **relative next-month returns across liquid US sector ETFs** with a leakage-aware walk-forward Ridge model, then converts those predictions into a dollar-neutral long/short portfolio.

## Research question
Can a simple, interpretable machine-learning model combine momentum, volatility and drawdown features to rank sector ETFs more robustly than a plain momentum rule?

## Universe and data
The experiment uses 11 liquid US sector ETFs (`XLB`, `XLC`, `XLE`, `XLF`, `XLI`, `XLK`, `XLP`, `XLRE`, `XLU`, `XLV`, `XLY`) downloaded from Yahoo Finance. `SPY` is retained as a market benchmark. Because `XLC` begins in 2018, the common sample starts in 2018.

## Features
All predictors are computed using information available at each month-end:
- 1-, 3-, 6- and 12-month momentum;
- 6- and 12-month momentum excluding the most recent month;
- 1- and 3-month realized volatility;
- 3-month drawdown from the rolling high.

The training label is the **next-month return relative to the contemporaneous cross-sectional mean**. This removes much of the common market move and makes the model focus on relative winners and losers.

## Walk-forward methodology
- Expanding-window validation: the model at month `t` is fit only on observations strictly before `t`.
- Minimum training history: 24 months.
- Model: `StandardScaler` + `Ridge(alpha=1.0)`.
- Portfolio: long the three highest predicted ETFs and short the three lowest.
- Risk weighting: inverse 3-month volatility within each side.
- Exposure: 50% gross long / 50% gross short, so the portfolio is dollar neutral before costs.
- Rebalancing: monthly.
- Transaction costs: 5 bps per unit of one-way turnover.
- Benchmark: a 6-month-minus-1-month momentum long/short rule evaluated on the exact same months, plus SPY for market context.
- Out-of-sample diagnostic: results are also reported separately for pre-2024 and 2024-present periods.

## Leakage controls
The project deliberately avoids random train/test shuffling. Each prediction is generated from an expanding training set containing only past months. Future returns are used only as historical labels after they would have become observable. The tests also verify market neutrality, target construction and accounting identities for gross returns, costs and net returns.

## Repository structure
- `src/strategy.py` — feature engineering, walk-forward Ridge model, portfolio construction and metrics.
- `run_backtest.py` — downloads real market data and runs the complete research pipeline.
- `tests/test_strategy.py` — unit tests for the panel, cross-sectional labels, neutrality and cost accounting.
- `notebooks/market_neutral_momentum.ipynb` — Google Colab-ready research notebook.
- `.github/workflows/ci.yml` — automatically runs tests, the real-data backtest and the notebook end-to-end.
- `results/` — generated metrics, predictions and figures from validated runs.

## Run locally
```bash
pip install -r requirements.txt
python -m pytest -q
python run_backtest.py
```

## What to explain in an interview
The model is intentionally simple. Ridge regression is not used because linear models are expected to dominate sophisticated hedge-fund systems; it is used because it gives a transparent test of whether several economically motivated features contain useful cross-sectional information. The main methodological point is the research process: strict time ordering, a realistic long/short construction, transaction costs, a simple benchmark, and separate out-of-sample diagnostics.

## Important limitation
This is an educational quantitative-research project, **not evidence of a profitable live trading system**. Performance depends on the ETF universe, period, transaction-cost assumptions and modelling choices. The model is deliberately small enough to audit and explain rather than optimized for maximum backtest performance.

## Author
Mohamed Atigui — MSc Mathematics & Artificial Intelligence, Université Paris-Saclay / CentraleSupélec
