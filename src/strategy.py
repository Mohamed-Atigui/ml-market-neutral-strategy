import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

PERIODS_PER_YEAR = 12

FEATURE_COLUMNS = [
    "mom1",
    "mom3",
    "mom6",
    "mom12",
    "mom6_ex1",
    "mom12_ex1",
    "vol1",
    "vol3",
    "dd3",
]


def monthly_end_dates(prices: pd.DataFrame) -> pd.DatetimeIndex:
    """Last available trading date of each calendar month."""
    return prices.groupby(prices.index.to_period("M")).tail(1).index


def build_monthly_panel(prices: pd.DataFrame) -> pd.DataFrame:
    """Create a leakage-safe cross-sectional ML panel from daily close prices."""
    prices = prices.sort_index().copy()
    daily_returns = prices.pct_change()
    dates = monthly_end_dates(prices)
    monthly_prices = prices.loc[dates]

    feature_frames = {
        "mom1": prices.pct_change(21).loc[dates],
        "mom3": prices.pct_change(63).loc[dates],
        "mom6": prices.pct_change(126).loc[dates],
        "mom12": prices.pct_change(252).loc[dates],
        "mom6_ex1": (prices.shift(21) / prices.shift(126) - 1.0).loc[dates],
        "mom12_ex1": (prices.shift(21) / prices.shift(252) - 1.0).loc[dates],
        "vol1": (daily_returns.rolling(21).std() * np.sqrt(252)).loc[dates],
        "vol3": (daily_returns.rolling(63).std() * np.sqrt(252)).loc[dates],
        "dd3": (prices / prices.rolling(63).max() - 1.0).loc[dates],
    }

    # Target is the next calendar month's asset return. This is used only as a
    # label for historical observations and never as an input feature.
    target = monthly_prices.shift(-1) / monthly_prices - 1.0

    rows = []
    for dt in dates:
        for asset in prices.columns:
            row = {"date": dt, "asset": asset, "target_return": target.loc[dt, asset]}
            for name, frame in feature_frames.items():
                row[name] = frame.loc[dt, asset]
            rows.append(row)

    panel = pd.DataFrame(rows).dropna().reset_index(drop=True)
    # Cross-sectional target strips the common market move from each training month.
    panel["target_relative"] = panel["target_return"] - panel.groupby("date")[
        "target_return"
    ].transform("mean")
    return panel


def _rank_to_weights(month: pd.DataFrame, score_col: str, n_long: int, n_short: int) -> pd.Series:
    """Construct a dollar-neutral, inverse-volatility long/short portfolio."""
    assets = month["asset"].tolist()
    weights = pd.Series(0.0, index=assets, dtype=float)

    longs = month.nlargest(n_long, score_col)["asset"].tolist()
    shorts = month.nsmallest(n_short, score_col)["asset"].tolist()
    vol = month.set_index("asset")["vol3"]

    inv_long = 1.0 / vol.loc[longs]
    inv_short = 1.0 / vol.loc[shorts]
    weights.loc[longs] = 0.5 * inv_long / inv_long.sum()
    weights.loc[shorts] = -0.5 * inv_short / inv_short.sum()
    return weights


def _portfolio_return(month: pd.DataFrame, weights: pd.Series) -> float:
    realized = month.set_index("asset")["target_return"]
    return float((weights * realized.reindex(weights.index)).sum())


def walk_forward_backtest(
    prices: pd.DataFrame,
    min_train_months: int = 24,
    alpha: float = 1.0,
    n_long: int = 3,
    n_short: int = 3,
    cost_bps: float = 5.0,
):
    """
    Expanding-window cross-sectional Ridge strategy.

    For every rebalance month, the model is trained only on prior months whose
    outcomes are already known. It then predicts relative next-month returns,
    goes long the top-ranked assets and short the bottom-ranked assets.
    """
    panel = build_monthly_panel(prices)
    dates = sorted(panel["date"].unique())
    assets = list(prices.columns)

    model = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
    previous_weights = pd.Series(0.0, index=assets)
    records = []
    prediction_rows = []

    for i, dt in enumerate(dates):
        if i < min_train_months:
            continue

        train = panel[panel["date"].isin(dates[:i])]
        current = panel[panel["date"] == dt].copy()

        model.fit(train[FEATURE_COLUMNS], train["target_relative"])
        current["prediction"] = model.predict(current[FEATURE_COLUMNS])

        weights = _rank_to_weights(current, "prediction", n_long, n_short).reindex(assets).fillna(0.0)
        gross_return = _portfolio_return(current, weights)
        turnover = float((weights - previous_weights).abs().sum())
        costs = turnover * cost_bps / 10000.0
        net_return = gross_return - costs

        current["weight"] = current["asset"].map(weights)
        prediction_rows.append(current)
        records.append(
            {
                "date": dt,
                "gross_return": gross_return,
                "net_return": net_return,
                "turnover": turnover,
                "cost": costs,
            }
        )
        previous_weights = weights

    returns = pd.DataFrame(records).set_index("date")
    predictions = pd.concat(prediction_rows, ignore_index=True)
    return {"returns": returns, "predictions": predictions, "panel": panel}


def momentum_baseline_backtest(
    prices: pd.DataFrame,
    evaluation_dates,
    n_long: int = 3,
    n_short: int = 3,
    cost_bps: float = 5.0,
):
    """Simple 6m-minus-1m momentum baseline evaluated on the same months."""
    panel = build_monthly_panel(prices)
    assets = list(prices.columns)
    previous_weights = pd.Series(0.0, index=assets)
    records = []

    for dt in evaluation_dates:
        current = panel[panel["date"] == dt].copy()
        weights = _rank_to_weights(current, "mom6_ex1", n_long, n_short).reindex(assets).fillna(0.0)
        gross_return = _portfolio_return(current, weights)
        turnover = float((weights - previous_weights).abs().sum())
        costs = turnover * cost_bps / 10000.0
        records.append(
            {
                "date": dt,
                "gross_return": gross_return,
                "net_return": gross_return - costs,
                "turnover": turnover,
                "cost": costs,
            }
        )
        previous_weights = weights

    return pd.DataFrame(records).set_index("date")


def performance_metrics(returns: pd.Series, turnover: pd.Series | None = None) -> dict:
    """Annualized metrics for monthly returns."""
    r = returns.dropna()
    if r.empty:
        raise ValueError("No returns available")

    cumulative = float((1.0 + r).prod())
    annualized_return = cumulative ** (PERIODS_PER_YEAR / len(r)) - 1.0
    annualized_volatility = float(r.std() * np.sqrt(PERIODS_PER_YEAR))
    sharpe = annualized_return / annualized_volatility if annualized_volatility > 0 else np.nan
    equity = (1.0 + r).cumprod()
    max_drawdown = float((equity / equity.cummax() - 1.0).min())

    metrics = {
        "annualized_return": float(annualized_return),
        "annualized_volatility": annualized_volatility,
        "sharpe_ratio": float(sharpe),
        "max_drawdown": max_drawdown,
        "cumulative_return": cumulative - 1.0,
    }
    if turnover is not None:
        aligned = turnover.reindex(r.index).fillna(0.0)
        metrics["average_monthly_turnover"] = float(aligned.mean())
        metrics["annualized_turnover"] = float(aligned.mean() * PERIODS_PER_YEAR)
    return metrics


def mean_rank_ic(predictions: pd.DataFrame) -> float:
    """Mean monthly Spearman rank correlation between prediction and realized return."""
    values = []
    for _, month in predictions.groupby("date"):
        values.append(month["prediction"].corr(month["target_return"], method="spearman"))
    return float(pd.Series(values).dropna().mean())
