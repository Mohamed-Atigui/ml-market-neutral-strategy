import numpy as np
import pandas as pd

from src.strategy import (
    FEATURE_COLUMNS,
    build_monthly_panel,
    performance_metrics,
    walk_forward_backtest,
)


def synthetic_prices(seed=7, n_days=900, n_assets=8):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2018-01-01", periods=n_days)
    common = rng.normal(0.0002, 0.007, n_days)
    idio = rng.normal(0.0, 0.010, (n_days, n_assets))
    rets = 0.30 * common[:, None] + idio
    px = 100 * np.exp(np.cumsum(rets, axis=0))
    return pd.DataFrame(px, index=dates, columns=[f"A{i}" for i in range(n_assets)])


def test_panel_contains_features_and_future_label():
    panel = build_monthly_panel(synthetic_prices())
    assert set(FEATURE_COLUMNS).issubset(panel.columns)
    assert {"target_return", "target_relative"}.issubset(panel.columns)
    assert not panel[FEATURE_COLUMNS + ["target_return"]].isna().any().any()


def test_relative_target_is_cross_sectionally_centered():
    panel = build_monthly_panel(synthetic_prices())
    monthly_means = panel.groupby("date")["target_relative"].mean()
    assert np.allclose(monthly_means.values, 0.0, atol=1e-12)


def test_walk_forward_weights_are_market_neutral():
    result = walk_forward_backtest(synthetic_prices(), min_train_months=12, n_long=2, n_short=2)
    predictions = result["predictions"]
    net_exposure = predictions.groupby("date")["weight"].sum()
    assert np.allclose(net_exposure.values, 0.0, atol=1e-10)


def test_transaction_costs_are_nonnegative():
    result = walk_forward_backtest(synthetic_prices(), min_train_months=12, cost_bps=5)
    returns = result["returns"]
    assert (returns["cost"] >= 0).all()
    assert np.allclose(returns["net_return"], returns["gross_return"] - returns["cost"])


def test_metrics_have_expected_fields():
    result = walk_forward_backtest(synthetic_prices(), min_train_months=12)
    m = performance_metrics(result["returns"]["net_return"], result["returns"]["turnover"])
    assert {
        "annualized_return",
        "annualized_volatility",
        "sharpe_ratio",
        "max_drawdown",
        "cumulative_return",
        "average_monthly_turnover",
        "annualized_turnover",
    }.issubset(m)
