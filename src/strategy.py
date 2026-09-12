import numpy as np
import pandas as pd

TRADING_DAYS = 252


def momentum_signal(prices: pd.DataFrame, lookback: int = 126, skip: int = 21) -> pd.DataFrame:
    """Medium-term momentum: return from t-lookback to t-skip."""
    if lookback <= skip:
        raise ValueError("lookback must be greater than skip")
    return prices.shift(skip) / prices.shift(lookback) - 1.0


def rolling_vol(returns: pd.DataFrame, window: int = 63) -> pd.DataFrame:
    """Annualized trailing volatility for each asset."""
    return returns.rolling(window).std() * np.sqrt(TRADING_DAYS)


def _rebalance_dates(index: pd.DatetimeIndex, frequency: str = "monthly") -> pd.DatetimeIndex:
    if frequency == "daily":
        return index
    if frequency != "monthly":
        raise ValueError("frequency must be 'daily' or 'monthly'")
    s = pd.Series(index=index, data=np.arange(len(index)))
    return s.groupby(index.to_period("M")).tail(1).index


def cross_sectional_weights(
    prices: pd.DataFrame,
    lookback: int = 126,
    skip: int = 21,
    vol_window: int = 63,
    long_frac: float = 0.25,
    short_frac: float = 0.25,
    rebalance_frequency: str = "monthly",
) -> pd.DataFrame:
    """
    Build dollar-neutral long/short weights.

    At each rebalance date, rank assets by medium-term momentum, go long the
    strongest names and short the weakest names, and inverse-volatility weight
    positions within each side. Between rebalances, weights are held constant.
    """
    if prices.shape[1] < 4:
        raise ValueError("At least four assets are required")

    returns = prices.pct_change()
    signal = momentum_signal(prices, lookback, skip)
    vol = rolling_vol(returns, vol_window).replace(0, np.nan)

    weights = pd.DataFrame(np.nan, index=prices.index, columns=prices.columns, dtype=float)
    rebal_dates = set(_rebalance_dates(prices.index, rebalance_frequency))

    for dt in prices.index:
        if dt not in rebal_dates:
            continue

        s = signal.loc[dt].dropna()
        v = vol.loc[dt].reindex(s.index).dropna()
        s = s.reindex(v.index)
        n = len(s)
        if n < 4:
            continue

        k_long = max(1, int(np.ceil(n * long_frac)))
        k_short = max(1, int(np.ceil(n * short_frac)))
        longs = s.nlargest(k_long).index
        shorts = s.nsmallest(k_short).index

        row = pd.Series(0.0, index=prices.columns)
        invv_l = (1.0 / v.loc[longs]).replace([np.inf, -np.inf], np.nan).dropna()
        invv_s = (1.0 / v.loc[shorts]).replace([np.inf, -np.inf], np.nan).dropna()

        if len(invv_l):
            row.loc[invv_l.index] = 0.5 * invv_l / invv_l.sum()
        if len(invv_s):
            row.loc[invv_s.index] = -0.5 * invv_s / invv_s.sum()
        weights.loc[dt] = row

    return weights.ffill().fillna(0.0)


def apply_vol_target(
    raw_strategy_returns: pd.Series,
    base_weights: pd.DataFrame,
    target_vol: float = 0.10,
    vol_window: int = 63,
    max_leverage: float = 2.0,
):
    """Scale portfolio exposure using only trailing realized strategy volatility."""
    realized = raw_strategy_returns.rolling(vol_window).std() * np.sqrt(TRADING_DAYS)
    leverage = (target_vol / realized.shift(1)).clip(lower=0.0, upper=max_leverage)
    leverage = leverage.replace([np.inf, -np.inf], np.nan).fillna(1.0)
    return base_weights.mul(leverage, axis=0), leverage


def backtest(
    prices: pd.DataFrame,
    lookback: int = 126,
    skip: int = 21,
    vol_window: int = 63,
    target_vol: float = 0.10,
    max_leverage: float = 2.0,
    cost_bps: float = 5.0,
    rebalance_frequency: str = "monthly",
):
    """Run a strictly lagged backtest with turnover-dependent transaction costs."""
    asset_returns = prices.pct_change().fillna(0.0)

    signal_weights = cross_sectional_weights(
        prices,
        lookback=lookback,
        skip=skip,
        vol_window=vol_window,
        rebalance_frequency=rebalance_frequency,
    )

    executed_weights = signal_weights.shift(1).fillna(0.0)
    raw_returns = (executed_weights * asset_returns).sum(axis=1)

    scaled_weights, leverage = apply_vol_target(
        raw_returns,
        executed_weights,
        target_vol=target_vol,
        vol_window=vol_window,
        max_leverage=max_leverage,
    )

    turnover = scaled_weights.diff().abs().sum(axis=1).fillna(0.0)
    gross_returns = (scaled_weights * asset_returns).sum(axis=1)
    costs = turnover * (cost_bps / 10000.0)
    net_returns = gross_returns - costs

    return {
        "weights": scaled_weights,
        "gross_returns": gross_returns,
        "net_returns": net_returns,
        "turnover": turnover,
        "costs": costs,
        "leverage": leverage,
    }


def max_drawdown(returns: pd.Series) -> float:
    equity = (1.0 + returns.fillna(0.0)).cumprod()
    drawdown = equity / equity.cummax() - 1.0
    return float(drawdown.min())


def performance_metrics(returns: pd.Series, turnover: pd.Series | None = None) -> dict:
    r = returns.dropna()
    if len(r) == 0:
        raise ValueError("No returns available")

    total = (1.0 + r).prod()
    ann_ret = total ** (TRADING_DAYS / len(r)) - 1.0
    ann_vol = r.std() * np.sqrt(TRADING_DAYS)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else np.nan

    out = {
        "annualized_return": float(ann_ret),
        "annualized_volatility": float(ann_vol),
        "sharpe_ratio": float(sharpe),
        "max_drawdown": max_drawdown(r),
        "cumulative_return": float(total - 1.0),
    }
    if turnover is not None:
        aligned = turnover.reindex(r.index).fillna(0.0)
        out["average_daily_turnover"] = float(aligned.mean())
        out["annualized_turnover"] = float(aligned.mean() * TRADING_DAYS)
    return out
