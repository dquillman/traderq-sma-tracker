"""
Portfolio module for TraderQ
Handles portfolio metrics calculation and backtesting strategies
"""

import pandas as pd
from datetime import date


# Constants
SMA_SHORT = 20
SMA_LONG = 200


def calculate_portfolio_metrics(portfolio: dict, start: date, end: date, mode: str,
                                load_data_func=None, _sma_func=None) -> dict:
    """
    Calculate portfolio-level metrics

    Args:
        portfolio: Dictionary with 'tickers' and 'weights'
        start: Start date for analysis
        end: End date for analysis
        mode: "Stocks" or "Crypto"
        load_data_func: Function to load ticker data (from data_loader module)
        _sma_func: SMA calculation function (from indicators module)
    """
    if not portfolio.get("tickers") or len(portfolio["tickers"]) == 0:
        return {"error": "No tickers in portfolio"}

    if not load_data_func:
        return {"error": "load_data function not provided"}

    tickers = portfolio["tickers"]
    weights = portfolio.get("weights", {})

    # Normalize weights to sum to 1.0
    total_weight = sum(weights.get(t, 1.0/len(tickers)) for t in tickers)
    if total_weight == 0:
        total_weight = len(tickers)

    portfolio_value = 0.0
    portfolio_return = 0.0
    individual_returns = {}

    for ticker in tickers:
        weight = weights.get(ticker, 1.0/len(tickers)) / total_weight if total_weight > 0 else 1.0/len(tickers)
        df = load_data_func(ticker, start, end, mode, data_source="Yahoo Finance")

        if df.empty:
            continue

        start_price = float(df.iloc[0]["close"])
        end_price = float(df.iloc[-1]["close"])
        ticker_return = ((end_price - start_price) / start_price) * 100

        portfolio_value += end_price * weight
        portfolio_return += ticker_return * weight
        individual_returns[ticker] = {
            "return": ticker_return,
            "weight": weight,
            "price": end_price
        }

    return {
        "portfolio_value": portfolio_value,
        "portfolio_return": portfolio_return,
        "individual_returns": individual_returns,
        "ticker_count": len(tickers)
    }


def backtest_strategy(df: pd.DataFrame, strategy: str = "golden_death",
                     initial_capital: float = 10000.0, _sma_func=None) -> dict:
    """
    Backtest a trading strategy. Returns performance metrics.

    Args:
        df: Price data DataFrame
        strategy: "golden_death" or "buy_hold"
        initial_capital: Starting capital
        _sma_func: SMA calculation function (from indicators module)
    """
    if df.empty or len(df) < SMA_LONG:
        return {"error": "Insufficient data for backtesting"}

    if not _sma_func:
        return {"error": "SMA function not provided"}

    df = df.copy()
    df["SMA20"] = _sma_func(df["close"], SMA_SHORT)
    df["SMA200"] = _sma_func(df["close"], SMA_LONG)

    capital = initial_capital
    shares = 0.0
    trades = []
    position = None  # "long" or None

    s_prev = df["SMA20"].shift(1)
    l_prev = df["SMA200"].shift(1)
    golden_crosses = (s_prev <= l_prev) & (df["SMA20"] > df["SMA200"])
    death_crosses = (s_prev >= l_prev) & (df["SMA20"] < df["SMA200"])

    for i in range(SMA_LONG, len(df)):
        price = float(df.iloc[i]["close"])
        date = df.index[i]

        if strategy == "golden_death":
            # Buy on Golden Cross, Sell on Death Cross
            if golden_crosses.iloc[i] and position is None:
                # Buy
                shares = capital / price
                capital = 0
                position = "long"
                trades.append({"date": date, "type": "BUY", "price": price, "shares": shares})

            elif death_crosses.iloc[i] and position == "long":
                # Sell
                capital = shares * price
                shares = 0
                position = None
                trades.append({"date": date, "type": "SELL", "price": price, "capital": capital})

        elif strategy == "buy_hold":
            # Buy and hold from start
            if i == SMA_LONG and position is None:
                shares = capital / price
                capital = 0
                position = "long"
                trades.append({"date": date, "type": "BUY", "price": price, "shares": shares})

    # Final value
    if position == "long" and shares > 0:
        final_price = float(df.iloc[-1]["close"])
        final_capital = shares * final_price
    else:
        final_capital = capital

    # Calculate metrics
    total_return = ((final_capital - initial_capital) / initial_capital) * 100
    num_trades = len([t for t in trades if t["type"] == "BUY"])

    # Calculate drawdown by tracking position through time
    equity_curve = []
    current_shares = 0.0
    current_capital = initial_capital
    current_position = None

    for i in range(SMA_LONG, len(df)):
        price = float(df.iloc[i]["close"])

        # Update position based on strategy
        if strategy == "golden_death":
            if golden_crosses.iloc[i] and current_position is None:
                current_shares = current_capital / price
                current_capital = 0
                current_position = "long"
            elif death_crosses.iloc[i] and current_position == "long":
                current_capital = current_shares * price
                current_shares = 0
                current_position = None
        elif strategy == "buy_hold":
            if i == SMA_LONG and current_position is None:
                current_shares = current_capital / price
                current_capital = 0
                current_position = "long"

        # Calculate current equity
        if current_position == "long":
            equity = current_shares * price
        else:
            equity = current_capital
        equity_curve.append(equity)

    peak = equity_curve[0]
    max_drawdown = 0.0
    for equity in equity_curve:
        if equity > peak:
            peak = equity
        drawdown = ((peak - equity) / peak) * 100
        if drawdown > max_drawdown:
            max_drawdown = drawdown

    return {
        "initial_capital": initial_capital,
        "final_capital": final_capital,
        "total_return": total_return,
        "num_trades": num_trades,
        "max_drawdown": max_drawdown,
        "trades": trades,
        "strategy": strategy
    }
