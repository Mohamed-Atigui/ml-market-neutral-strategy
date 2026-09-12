# Market-Neutral Momentum Strategy

[![Quant research CI](https://github.com/Mohamed-Atigui/ml-market-neutral-strategy/actions/workflows/ci.yml/badge.svg)](https://github.com/Mohamed-Atigui/ml-market-neutral-strategy/actions/workflows/ci.yml)

A reproducible quantitative research project implementing a **cross-sectional market-neutral momentum strategy** with volatility scaling, transaction costs and strict temporal lagging to avoid look-ahead bias.

## Research question
Can medium-term relative momentum across liquid US sector ETFs produce a robust long/short signal after implementation costs?

## Methodology
- **Universe:** 11 liquid US sector ETFs.
- **Signal:** 6-month momentum excluding the most recent month (126 trading days minus 21 days).
- **Rebalancing:** monthly.
- **Portfolio:** long the top quartile and short the bottom quartile of the cross-section.
- **Risk weighting:** inverse-volatility weighting inside the long and short books.
- **Exposure:** approximately 50% long / 50% short before volatility targeting.
- **Volatility target:** 10% annualized, estimated only from trailing strategy returns, with leverage capped at 2x.
- **Execution:** portfolio weights are shifted by one trading day before returns are applied.
- **Costs:** 5 bps per unit of one-way turnover.
- **Evaluation:** annualized return, annualized volatility, Sharpe ratio, maximum drawdown, turnover and cumulative return.
- **Robustness:** chronological 2017–2023 / 2024–present split and sensitivity tests across lookback horizons and transaction costs.

## Why this is not a toy backtest
The workflow explicitly separates signal formation from execution, prevents future information from entering portfolio weights, includes turnover-dependent costs, compares the strategy to SPY and checks out-of-sample behaviour instead of relying on a single in-sample result.

## Repository structure
- `src/strategy.py` — signal construction, portfolio weighting, risk targeting and metrics.
- `run_backtest.py` — downloads real market data and runs the full experiment.
- `tests/test_strategy.py` — unit tests for neutrality, exposure, costs and output integrity.
- `notebooks/market_neutral_momentum.ipynb` — Google Colab-ready research notebook.
- `.github/workflows/ci.yml` — automated tests, real-data backtest and notebook execution.
- `results/` — generated metrics and figures from the latest validated run.

## Run locally
```bash
pip install -r requirements.txt
pytest -q
python run_backtest.py
```

## Interview-level explanation
The strategy ranks sector ETFs using medium-term momentum, goes long recent relative winners and short recent relative losers, then scales individual positions by inverse volatility so that high-volatility assets do not dominate portfolio risk. The portfolio is rebalanced monthly. Every target weight is delayed by one trading day before being multiplied by asset returns, preventing look-ahead bias. Transaction costs are proportional to turnover, and a trailing volatility estimate is used to scale overall exposure toward a 10% annualized target.

## Important limitation
This repository is a research project, **not** evidence of a profitable live trading system. Results depend on the investment universe, sample period, transaction-cost assumptions and modelling choices.

## Author
Mohamed Atigui — MSc Mathematics & Artificial Intelligence, Université Paris-Saclay / CentraleSupélec
