import numpy as np
import pandas as pd

from src.strategy import backtest, cross_sectional_weights, performance_metrics


def synthetic_prices(seed=7, n_days=700, n_assets=8):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-01", periods=n_days)
    common = rng.normal(0.0002, 0.007, n_days)
    idio = rng.normal(0.0, 0.010, (n_days, n_assets))
    rets = 0.30 * common[:, None] + idio
    px = 100 * np.exp(np.cumsum(rets, axis=0))
    return pd.DataFrame(px, index=dates, columns=[f"A{i}" for i in range(n_assets)])


def test_weights_are_dollar_neutral_after_warmup():
    prices = synthetic_prices()
    w = cross_sectional_weights(prices, rebalance_frequency="monthly")
    active = w.abs().sum(axis=1) > 0
    net = w.loc[active].sum(axis=1)
    assert np.allclose(net.values, 0.0, atol=1e-10)


def test_gross_exposure_is_one_before_vol_target():
    prices = synthetic_prices()
    w = cross_sectional_weights(prices, rebalance_frequency="monthly")
    active = w.abs().sum(axis=1) > 0
    gross = w.loc[active].abs().sum(axis=1)
    assert np.allclose(gross.values, 1.0, atol=1e-10)


def test_backtest_outputs_are_finite():
    prices = synthetic_prices()
    bt = backtest(prices, rebalance_frequency="monthly")
    assert np.isfinite(bt["net_returns"]).all()
    assert np.isfinite(bt["turnover"]).all()
    assert (bt["costs"] >= 0).all()


def test_transaction_costs_reduce_terminal_wealth():
    prices = synthetic_prices()
    no_cost = backtest(prices, cost_bps=0, rebalance_frequency="monthly")
    with_cost = backtest(prices, cost_bps=10, rebalance_frequency="monthly")
    wealth_no_cost = (1 + no_cost["net_returns"]).prod()
    wealth_with_cost = (1 + with_cost["net_returns"]).prod()
    assert wealth_with_cost <= wealth_no_cost + 1e-12


def test_metrics_have_expected_fields():
    prices = synthetic_prices()
    bt = backtest(prices, rebalance_frequency="monthly")
    m = performance_metrics(bt["net_returns"], bt["turnover"])
    expected = {
        "annualized_return",
        "annualized_volatility",
        "sharpe_ratio",
        "max_drawdown",
        "cumulative_return",
        "average_daily_turnover",
        "annualized_turnover",
    }
    assert expected.issubset(m)
