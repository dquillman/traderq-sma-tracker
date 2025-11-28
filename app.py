# TraderQ — SMA 20/200 Tracker (Stocks + Crypto)
# v1.4.8
# Single-file Streamlit app with clean SMA logic, pretouch screener, cross markers,
# crypto fallback, and trend chips (Bullish/Bearish) without emoji.

# CRITICAL: Write to stderr immediately to track startup
import sys
sys.stderr.write("=" * 70 + "\n")
sys.stderr.write("APP.PY: Starting imports...\n")
sys.stderr.flush()

from __future__ import annotations

sys.stderr.write("✓ __future__ imported\n")
sys.stderr.flush()

import json
import math
import time
from datetime import date, datetime, timedelta
from functools import lru_cache

sys.stderr.write("✓ Standard library imports done\n")
sys.stderr.flush()

import numpy as np
import pandas as pd
import plotly.graph_objects as go

sys.stderr.write("✓ Data science imports done\n")
sys.stderr.flush()


def add_cross_markers(fig: go.Figure, df: pd.DataFrame,
                      price_col: str = "Close",
                      s20: str = "SMA20",
                      s200: str = "SMA200",
                      row: int = 1, col: int = 1) -> None:
    """
    Golden cross: 20 crosses above 200 (green triangle-up)
    Death  cross: 20 crosses below 200 (red triangle-down)
    Works if df has df[price_col], df[s20], df[s200].
    """
    if df is None or df.empty:
        return
    needed = (price_col, s20, s200)
    if any(c not in df.columns for c in needed):
        return
    s_prev = df[s20].shift(1)
    l_prev = df[s200].shift(1)
    cross_up = (s_prev <= l_prev) & (df[s20] > df[s200])
    cross_dn = (s_prev >= l_prev) & (df[s20] < df[s200])
    xu, yu = df.index[cross_up], df.loc[cross_up, price_col]
    xd, yd = df.index[cross_dn], df.loc[cross_dn, price_col]
    if len(xu):
        fig.add_trace(go.Scatter(
            x=xu, y=yu, mode="markers", name="Golden Cross",
            marker=dict(symbol="triangle-up", size=25, color="#17c964",
                        line=dict(width=1, color="#0b3820")),
            hovertemplate="Golden: %{x|%Y-%m-%d}<br>Price: %{y:.2f}<extra></extra>",
            showlegend=True
        ), row=row, col=col)
    if len(xd):
        fig.add_trace(go.Scatter(
            x=xd, y=yd, mode="markers", name="Death Cross",
            marker=dict(symbol="triangle-down", size=25, color="#f31260",
                        line=dict(width=1, color="#4a0b19")),
            hovertemplate="Death: %{x|%Y-%m-%d}<br>Price: %{y:.2f}<extra></extra>",
            showlegend=True
        ), row=row, col=col)


sys.stderr.write("Importing streamlit...\n")
sys.stderr.flush()

import streamlit as st

sys.stderr.write("✓ Streamlit imported\n")
sys.stderr.flush()

# Import patches - wrap in try/except to prevent crashes
sys.stderr.write("Importing ui_glow_patch...\n")
sys.stderr.flush()
try:
    import ui_glow_patch
    sys.stderr.write("✓ ui_glow_patch imported\n")
    sys.stderr.flush()
except Exception as e:
    sys.stderr.write(f"✗ Could not import ui_glow_patch: {e}\n")
    sys.stderr.flush()
    import traceback
    traceback.print_exc(file=sys.stderr)
    sys.stderr.flush()
    ui_glow_patch = None

sys.stderr.write("Importing yf_patch...\n")
sys.stderr.flush()
try:
    import yf_patch  # glow+session patch
    sys.stderr.write("✓ yf_patch imported\n")
    sys.stderr.flush()
except Exception as e:
    sys.stderr.write(f"✗ Could not import yf_patch: {e}\n")
    sys.stderr.flush()
    import traceback
    traceback.print_exc(file=sys.stderr)
    sys.stderr.flush()
    yf_patch = None

sys.stderr.write("All imports complete. Starting app code...\n")
sys.stderr.flush()

# Firebase imports - lazy loaded to avoid import-time errors
# These will be imported only when needed, after st.set_page_config
firebase_auth = None
firebase_db = None

APP_VERSION = "v2.6.0"
# v2.6.0 – Complete Firebase conversion: All data storage migrated to Firestore, multi-user support, cloud deployment ready
# v2.5.0 – Added Extended MACD indicator with adjustable sideways detection (lookback bars and range threshold), flattens MACD during range-bound markets
# v2.4.0 – Added Fair Value Gap indicator, indicator selection for AI Recommendations, trade parameter visualization (red risk/green reward zones)
# v2.3.0 – Added Supertrend and EMA indicators, integrated Supertrend into AI Recommendations
# v2.2.0 – AI Recommendations system with comprehensive analysis (SMA, RSI, News Sentiment)
# v2.1.0 – risk management dashboard, signal generator, win rate analytics, position tracking, market regime detection

# --- Settings / Defaults ---
DEFAULT_STOCKS = ["^GSPC", "^DJI", "^IXIC", "SPY", "QQQ"]
DEFAULT_CRYPTOS = ["BTC-USD", "ETH-USD", "SOL-USD"]
DEFAULT_PERIOD_DAYS = 365 * 2  # 2 years for stocks; crypto API may cap to 365 in fallback
SMA_SHORT = 20
SMA_LONG = 200

# Global variables for Firebase (initialized in main execution)
auth = None
user_id = None
db = None

def load_custom_tickers() -> dict:
    """Load custom tickers from Firestore."""
    try:
        # user_id and db are global variables set after authentication
        if user_id is None or db is None:
            return {"custom": {"Stocks": [], "Crypto": []}, "selected": {"Stocks": [], "Crypto": []}}
        data = db.get_custom_tickers(user_id)
        # Ensure both custom and selected keys exist
        if "custom" not in data:
            data["custom"] = {"Stocks": [], "Crypto": []}
        if "selected" not in data:
            data["selected"] = {"Stocks": [], "Crypto": []}
        return data
    except Exception:
        return {"custom": {"Stocks": [], "Crypto": []}, "selected": {"Stocks": [], "Crypto": []}}

def save_custom_tickers(data: dict):
    """Save custom tickers to Firestore."""
    try:
        # user_id and db are global variables set after authentication
        if user_id is None or db is None:
            st.error("Not authenticated. Please log in.")
            return
        db.save_custom_tickers(user_id, data)
    except Exception as e:
        st.error(f"Failed to save custom tickers: {e}")

# --- Alerts persistence ---
def load_alerts() -> list:
    """Load alerts from Firestore."""
    try:
        if user_id is None or db is None:
            return []
        return db.get_alerts(user_id)
    except Exception:
        return []

def save_alerts(alerts: list):
    """Save alerts to Firestore."""
    try:
        if user_id is None or db is None:
            st.error("Not authenticated. Please log in.")
            return
        db.save_alerts(user_id, alerts)
    except Exception as e:
        st.error(f"Failed to save alerts: {e}")

# --- Portfolio persistence ---
def load_portfolio() -> dict:
    """Load portfolio from Firestore."""
    try:
        if user_id is None or db is None:
            return {"tickers": [], "weights": {}}
        return db.get_portfolio(user_id)
    except Exception:
        return {"tickers": [], "weights": {}}

def save_portfolio(portfolio: dict):
    """Save portfolio to Firestore."""
    try:
        if user_id is None or db is None:
            st.error("Not authenticated. Please log in.")
            return
        db.save_portfolio(user_id, portfolio)
    except Exception as e:
        st.error(f"Failed to save portfolio: {e}")

# --- Cross history persistence ---
def load_cross_history() -> dict:
    """Load cross history from Firestore."""
    try:
        if user_id is None or db is None:
            return {}
        return db.get_cross_history(user_id)
    except Exception:
        return {}

def save_cross_history(history: dict):
    """Save cross history to Firestore."""
    try:
        if user_id is None or db is None:
            st.error("Not authenticated. Please log in.")
            return
        db.save_cross_history(user_id, history)
    except Exception as e:
        st.error(f"Failed to save cross history: {e}")

# --- Watchlists persistence ---
def load_watchlists() -> dict:
    """Load watchlists from Firestore."""
    try:
        if user_id is None or db is None:
            return {}
        return db.get_watchlists(user_id)
    except Exception:
        return {}

def save_watchlists(watchlists: dict):
    """Save watchlists to Firestore."""
    try:
        if user_id is None or db is None:
            st.error("Not authenticated. Please log in.")
            return
        db.save_watchlists(user_id, watchlists)
    except Exception as e:
        st.error(f"Failed to save watchlists: {e}")

# --- Performance Metrics ---
def calculate_performance_metrics(df: pd.DataFrame) -> dict:
    """Calculate performance metrics: returns, Sharpe ratio, volatility."""
    if df.empty or "close" not in df.columns:
        return {}
    
    returns = df["close"].pct_change().dropna()
    if returns.empty:
        return {}
    
    # Total return
    total_return = (df["close"].iloc[-1] / df["close"].iloc[0] - 1) * 100
    
    # Annualized return (assuming daily data)
    days = len(df)
    annualized_return = ((df["close"].iloc[-1] / df["close"].iloc[0]) ** (252 / days) - 1) * 100 if days > 0 else 0
    
    # Volatility (annualized)
    volatility = returns.std() * np.sqrt(252) * 100
    
    # Sharpe ratio (assuming risk-free rate of 0 for simplicity)
    sharpe_ratio = (annualized_return / 100) / (volatility / 100) if volatility > 0 else 0
    
    # Maximum drawdown
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = drawdown.min() * 100
    
    # Win rate (if we consider up days as wins)
    win_rate = (returns > 0).sum() / len(returns) * 100 if len(returns) > 0 else 0
    
    return {
        "total_return": total_return,
        "annualized_return": annualized_return,
        "volatility": volatility,
        "sharpe_ratio": sharpe_ratio,
        "max_drawdown": max_drawdown,
        "win_rate": win_rate,
        "days": days
    }

# --- Email Configuration & Alerts ---
def load_email_config() -> dict:
    """Load email configuration from Firestore."""
    try:
        if user_id is None or db is None:
            return {}
        return db.get_email_config(user_id)
    except Exception:
        return {}

def save_email_config(config: dict):
    """Save email configuration to Firestore."""
    try:
        if user_id is None or db is None:
            st.error("Not authenticated. Please log in.")
            return
        db.save_email_config(user_id, config)
    except Exception as e:
        st.error(f"Failed to save email config: {e}")

# --- Kraken API Configuration ---
def load_kraken_config() -> dict:
    """Load Kraken API configuration from Firestore."""
    try:
        if user_id is None or db is None:
            return {}
        return db.get_kraken_config(user_id)
    except Exception:
        return {}

def save_kraken_config(config: dict):
    """Save Kraken API configuration to Firestore."""
    try:
        if user_id is None or db is None:
            st.error("Not authenticated. Please log in.")
            return
        db.save_kraken_config(user_id, config)
    except Exception as e:
        st.error(f"Failed to save Kraken config: {e}")

# --- Kraken API Functions ---
def get_kraken_client(api_key: str = None, api_secret: str = None):
    """Initialize Kraken API client."""
    # Try python-kraken-sdk first (newer SDK)
    try:
        from kraken.spot import SpotClient, User, Trade
    except ImportError:
        # Fallback to krakenex (older library)
        try:
            import krakenex
        except ImportError:
            st.error("⚠️ Kraken SDK not installed. Please run: `pip install python-kraken-sdk` or `pip install krakenex`")
            return None
        
        # Use krakenex
        try:
            if not api_key or not api_secret:
                config = load_kraken_config()
                api_key = config.get("api_key", "")
                api_secret = config.get("api_secret", "")
            
            if not api_key or not api_secret:
                return None
            
            api = krakenex.API(key=api_key, secret=api_secret)
            # Wrap in a compatible interface
            class KrakenClientWrapper:
                def __init__(self, api):
                    self.api = api
                def get_account_balance(self):
                    return self.api.query_private('Balance')
                def get_open_orders(self):
                    return self.api.query_private('OpenOrders')
                def add_order(self, **kwargs):
                    return self.api.query_private('AddOrder', kwargs)
                def cancel_order(self, txid=None):
                    return self.api.query_private('CancelOrder', {'txid': txid})
            return KrakenClientWrapper(api)
        except Exception as e:
            st.error(f"⚠️ Error initializing Kraken client (krakenex): {str(e)}")
            return None
    
    # Use python-kraken-sdk
    try:
        if not api_key or not api_secret:
            config = load_kraken_config()
            api_key = config.get("api_key", "")
            api_secret = config.get("api_secret", "")
        
        if not api_key or not api_secret:
            return None
        
        # Create SpotClient and wrap User/Trade classes for compatibility
        client = SpotClient(key=api_key, secret=api_secret)
        user = User(client)
        trade = Trade(client)
        
        # Create a wrapper class to match the expected interface
        class KrakenClientWrapper:
            def __init__(self, client, user, trade):
                self.client = client
                self.user = user
                self.trade = trade
            
            def get_account_balance(self):
                return self.user.get_account_balance()
            
            def get_open_orders(self):
                return self.user.get_open_orders()
            
            def add_order(self, **kwargs):
                # Convert to python-kraken-sdk format
                # create_order expects: ordertype, side, pair, volume, price (optional)
                order_params = {
                    "ordertype": kwargs.get("ordertype"),  # "market" or "limit"
                    "side": kwargs.get("type"),  # "buy" or "sell"
                    "pair": kwargs.get("pair"),
                    "volume": kwargs.get("volume")
                }
                if "price" in kwargs and kwargs["price"]:
                    order_params["price"] = kwargs["price"]
                return self.trade.create_order(**order_params)
            
            def cancel_order(self, txid=None):
                return self.trade.cancel_order(txid=txid)
        
        return KrakenClientWrapper(client, user, trade)
    except Exception as e:
        st.error(f"⚠️ Error initializing Kraken client: {str(e)}\n\nThis might be due to:\n- Invalid API key format\n- Network connectivity issues\n- API key permissions")
        return None

def get_kraken_balance(client) -> dict:
    """Get account balance from Kraken."""
    try:
        if not client:
            return {"error": "Client not initialized"}
        result = client.get_account_balance()
        # Handle different response formats
        if isinstance(result, dict):
            if "error" in result and result["error"]:
                # Kraken returns error as a list - empty list means no error
                if isinstance(result["error"], list) and len(result["error"]) > 0:
                    error_msg = ", ".join(result["error"])
                    return {"error": error_msg}
                elif not isinstance(result["error"], list) and result["error"]:
                    return {"error": str(result["error"])}
            # Extract balance from result format
            if "result" in result:
                balance = result["result"]
                # Kraken balance format: {"result": {"ZUSD": "100.00", "XXBT": "0.5"}}
                return {"success": True, "balance": balance}
            elif "balance" in result:
                return {"success": True, "balance": result["balance"]}
            else:
                return {"success": True, "balance": result}
        return {"success": True, "balance": result}
    except Exception as e:
        return {"error": str(e)}

def get_kraken_open_orders(client) -> dict:
    """Get open orders from Kraken."""
    try:
        if not client:
            return {"error": "Client not initialized"}
        result = client.get_open_orders()
        # Handle different response formats
        if isinstance(result, dict):
            if "error" in result and result["error"]:
                # Kraken returns error as a list - empty list means no error
                if isinstance(result["error"], list) and len(result["error"]) > 0:
                    error_msg = ", ".join(result["error"])
                    return {"error": error_msg}
                elif not isinstance(result["error"], list) and result["error"]:
                    return {"error": str(result["error"])}
            # Extract orders from result format
            if "result" in result:
                orders = result["result"]
                # Kraken orders format: {"result": {"open": {...}}}
                if "open" in orders:
                    orders = orders["open"]
                return {"success": True, "orders": orders}
            elif "orders" in result:
                return {"success": True, "orders": result["orders"]}
            else:
                return {"success": True, "orders": result}
        return {"success": True, "orders": result}
    except Exception as e:
        return {"error": str(e)}

def place_kraken_order(client, pair: str, type: str, ordertype: str, volume: float, price: float = None, dry_run: bool = False) -> dict:
    """
    Place an order on Kraken.
    
    Args:
        client: Kraken API client
        pair: Trading pair (e.g., "XBTUSD", "ETHUSD")
        type: "buy" or "sell"
        ordertype: "market", "limit", "stop-loss", etc.
        volume: Order volume
        price: Limit price (required for limit orders)
        dry_run: If True, validate order without placing it
    
    Returns:
        dict with order result or error
    """
    try:
        if not client:
            return {"error": "Client not initialized"}
        
        if dry_run:
            # Just validate the order parameters
            if ordertype == "limit" and price is None:
                return {"error": "Limit price required for limit orders"}
            if volume <= 0:
                return {"error": "Volume must be greater than 0"}
            return {"success": True, "dry_run": True, "message": "Order parameters validated (dry run)"}
        
        # Build order parameters (format depends on SDK)
        order_params = {
            "pair": pair,
            "type": type,  # "buy" or "sell"
            "ordertype": ordertype,  # "market" or "limit"
            "volume": str(volume)
        }
        
        if ordertype == "limit" and price:
            order_params["price"] = str(price)
        
        result = client.add_order(**order_params)
        
        # Handle different response formats
        if isinstance(result, dict):
            if "error" in result and result["error"]:
                error_msg = result["error"] if isinstance(result["error"], list) else [str(result["error"])]
                return {"error": ", ".join(error_msg)}
            if "result" in result:
                order_data = result["result"]
            else:
                order_data = result
            return {"success": True, "order": order_data}
        
        return {"success": True, "order": result}
    except Exception as e:
        return {"error": str(e)}

def cancel_kraken_order(client, txid: str) -> dict:
    """Cancel an order on Kraken."""
    try:
        if not client:
            return {"error": "Client not initialized"}
        result = client.cancel_order(txid=txid)
        # Handle different response formats
        if isinstance(result, dict):
            if "error" in result and result["error"]:
                error_msg = result["error"] if isinstance(result["error"], list) else [str(result["error"])]
                return {"error": ", ".join(error_msg)}
            if "result" in result:
                cancel_data = result["result"]
            else:
                cancel_data = result
            return {"success": True, "result": cancel_data}
        return {"success": True, "result": result}
    except Exception as e:
        return {"error": str(e)}

def get_kraken_tradable_pairs(client) -> list:
    """Get list of tradable pairs on Kraken."""
    try:
        if not client:
            return []
        # Use public endpoint for asset pairs
        import requests
        response = requests.get("https://api.kraken.com/0/public/AssetPairs")
        if response.status_code == 200:
            data = response.json()
            if "result" in data:
                pairs = list(data["result"].keys())
                # Filter for common crypto pairs
                common_pairs = [p for p in pairs if any(c in p.upper() for c in ["USD", "EUR", "USDT"])]
                return sorted(common_pairs)[:50]  # Return first 50
        return []
    except Exception:
        # Return common pairs as fallback
        return ["XBTUSD", "ETHUSD", "ADAUSD", "SOLUSD", "DOTUSD", "LINKUSD", "MATICUSD", "AVAXUSD"]

def convert_ticker_to_kraken_pair(ticker: str) -> str:
    """Convert ticker format (e.g., BTC-USD) to Kraken pair format (e.g., XBTUSD)."""
    mapping = {
        "BTC-USD": "XBTUSD",
        "ETH-USD": "ETHUSD",
        "ADA-USD": "ADAUSD",
        "SOL-USD": "SOLUSD",
        "DOT-USD": "DOTUSD",
        "LINK-USD": "LINKUSD",
        "MATIC-USD": "MATICUSD",
        "AVAX-USD": "AVAXUSD",
        "DOGE-USD": "XDGUSD",
        "XRP-USD": "XRPUSD",
        "LTC-USD": "LTCUSD",
    }
    
    ticker_upper = ticker.upper()
    if ticker_upper in mapping:
        return mapping[ticker_upper]
    
    # Try to convert generically
    ticker_clean = ticker_upper.replace("-USD", "USD").replace("-", "")
    # Kraken uses XBT for Bitcoin
    if ticker_clean.startswith("BTC"):
        ticker_clean = ticker_clean.replace("BTC", "XBT")
    return ticker_clean

def validate_bracket_prices(side: str, entry_price: float, stop_loss_price: float, take_profit_price: float) -> dict:
    """
    Validate bracket order price levels make sense.
    
    For BUY orders:
        - Stop-loss should be BELOW entry
        - Take-profit should be ABOVE entry
    
    For SELL orders:
        - Stop-loss should be ABOVE entry
        - Take-profit should be BELOW entry
    
    Returns:
        dict with "valid": bool and "error": str if invalid
    """
    if not entry_price or entry_price <= 0:
        return {"valid": False, "error": "Entry price must be greater than 0"}
    
    if not stop_loss_price or stop_loss_price <= 0:
        return {"valid": False, "error": "Stop-loss price must be greater than 0"}
    
    if not take_profit_price or take_profit_price <= 0:
        return {"valid": False, "error": "Take-profit price must be greater than 0"}
    
    side_lower = side.lower()
    
    if side_lower == "buy":
        # For buy orders: SL below entry, TP above entry
        if stop_loss_price >= entry_price:
            return {"valid": False, "error": "For BUY orders, stop-loss must be BELOW entry price"}
        if take_profit_price <= entry_price:
            return {"valid": False, "error": "For BUY orders, take-profit must be ABOVE entry price"}
    elif side_lower == "sell":
        # For sell orders: SL above entry, TP below entry
        if stop_loss_price <= entry_price:
            return {"valid": False, "error": "For SELL orders, stop-loss must be ABOVE entry price"}
        if take_profit_price >= entry_price:
            return {"valid": False, "error": "For SELL orders, take-profit must be BELOW entry price"}
    else:
        return {"valid": False, "error": "Side must be 'buy' or 'sell'"}
    
    return {"valid": True}

def calculate_bracket_metrics(entry_price: float, stop_loss_price: float, take_profit_price: float, volume: float) -> dict:
    """
    Calculate risk/reward metrics for bracket order.
    
    Returns:
        dict with risk/reward analysis
    """
    risk_per_unit = abs(entry_price - stop_loss_price)
    reward_per_unit = abs(take_profit_price - entry_price)
    
    total_risk = risk_per_unit * volume
    total_reward = reward_per_unit * volume
    
    risk_pct = (risk_per_unit / entry_price) * 100
    reward_pct = (reward_per_unit / entry_price) * 100
    
    risk_reward_ratio = reward_per_unit / risk_per_unit if risk_per_unit > 0 else 0
    
    return {
        "risk_per_unit": risk_per_unit,
        "reward_per_unit": reward_per_unit,
        "total_risk": total_risk,
        "total_reward": total_reward,
        "risk_pct": risk_pct,
        "reward_pct": reward_pct,
        "risk_reward_ratio": risk_reward_ratio
    }

def place_bracket_order(client, pair: str, side: str, volume: float, 
                       entry_price: float = None, 
                       stop_loss_price: float = None, 
                       take_profit_price: float = None,
                       dry_run: bool = False) -> dict:
    """
    Place a bracket order: entry + stop-loss + take-profit.
    
    Args:
        client: Kraken API client
        pair: Trading pair (e.g., "XBTUSD")
        side: "buy" or "sell"
        volume: Order volume
        entry_price: Entry price (None for market order)
        stop_loss_price: Stop-loss trigger price
        take_profit_price: Take-profit limit price
        dry_run: If True, validate without placing
    
    Returns:
        dict with results for all three orders
    """
    try:
        if not client:
            return {"error": "Client not initialized"}
        
        # Determine entry order type
        entry_ordertype = "market" if entry_price is None else "limit"
        effective_entry_price = entry_price if entry_price else 0  # Will be filled at market price
        
        # Validate bracket prices (skip if market entry, use current market price estimate)
        if entry_price:
            validation = validate_bracket_prices(side, entry_price, stop_loss_price, take_profit_price)
            if not validation["valid"]:
                return {"error": validation["error"]}
        
        if dry_run:
            # Validate all parameters
            if volume <= 0:
                return {"error": "Volume must be greater than 0"}
            if entry_ordertype == "limit" and not entry_price:
                return {"error": "Entry price required for limit orders"}
            if not stop_loss_price or stop_loss_price <= 0:
                return {"error": "Valid stop-loss price required"}
            if not take_profit_price or take_profit_price <= 0:
                return {"error": "Valid take-profit price required"}
            
            return {
                "success": True,
                "dry_run": True,
                "message": "Bracket order validated successfully",
                "entry_type": entry_ordertype,
                "stop_loss_type": "stop-loss-limit",
                "take_profit_type": "take-profit-limit"
            }
        
        results = {
            "entry_order": None,
            "stop_loss_order": None,
            "take_profit_order": None,
            "errors": []
        }
        
        # 1. Place entry order
        entry_result = place_kraken_order(
            client,
            pair=pair,
            type=side,
            ordertype=entry_ordertype,
            volume=volume,
            price=entry_price,
            dry_run=False
        )
        
        if "error" in entry_result:
            return {"error": f"Entry order failed: {entry_result['error']}"}
        
        results["entry_order"] = entry_result
        
        # 2. Place stop-loss order
        # For Kraken, we use stop-loss-limit order type
        # The stop-loss triggers at stop_loss_price, then places a limit order slightly worse to ensure fill
        stop_offset = abs(stop_loss_price * 0.005)  # 0.5% offset for limit price
        stop_limit_price = stop_loss_price - stop_offset if side == "buy" else stop_loss_price + stop_offset
        
        try:
            # Kraken stop-loss orders use opposite side (if you bought, you sell to exit)
            stop_side = "sell" if side == "buy" else "buy"
            
            stop_params = {
                "pair": pair,
                "type": stop_side,
                "ordertype": "stop-loss-limit",
                "volume": str(volume),
                "price": str(stop_loss_price),  # Trigger price
                "price2": str(stop_limit_price)  # Limit price after trigger
            }
            
            stop_result = client.add_order(**stop_params)
            
            if isinstance(stop_result, dict):
                if "error" in stop_result and stop_result["error"]:
                    error_msg = stop_result["error"] if isinstance(stop_result["error"], list) else [str(stop_result["error"])]
                    results["errors"].append(f"Stop-loss failed: {', '.join(error_msg)}")
                else:
                    results["stop_loss_order"] = stop_result
            else:
                results["stop_loss_order"] = stop_result
                
        except Exception as e:
            results["errors"].append(f"Stop-loss error: {str(e)}")
        
        # 3. Place take-profit order
        try:
            # Take-profit also uses opposite side
            tp_side = "sell" if side == "buy" else "buy"
            
            tp_params = {
                "pair": pair,
                "type": tp_side,
                "ordertype": "take-profit-limit",
                "volume": str(volume),
                "price": str(take_profit_price),  # Trigger price
                "price2": str(take_profit_price)  # Limit price (same as trigger for take-profit)
            }
            
            tp_result = client.add_order(**tp_params)
            
            if isinstance(tp_result, dict):
                if "error" in tp_result and tp_result["error"]:
                    error_msg = tp_result["error"] if isinstance(tp_result["error"], list) else [str(tp_result["error"])]
                    results["errors"].append(f"Take-profit failed: {', '.join(error_msg)}")
                else:
                    results["take_profit_order"] = tp_result
            else:
                results["take_profit_order"] = tp_result
                
        except Exception as e:
            results["errors"].append(f"Take-profit error: {str(e)}")
        
        # Determine overall success
        if results["entry_order"] and results["stop_loss_order"] and results["take_profit_order"]:
            results["success"] = True
            results["message"] = "All three orders placed successfully!"
        elif results["entry_order"] and results["stop_loss_order"]:
            results["success"] = True
            results["message"] = "Entry and stop-loss placed. Take-profit failed - please place manually!"
        elif results["entry_order"]:
            results["success"] = True
            results["message"] = "⚠️ WARNING: Only entry order placed! Stop-loss and take-profit failed - PLACE MANUALLY IMMEDIATELY!"
        else:
            results["success"] = False
            results["message"] = "Bracket order failed"
        
        return results
        
    except Exception as e:
        return {"error": str(e)}

def calculate_easy_mode_signal(df: pd.DataFrame) -> dict:
    """
    Calculate a simplified 'Traffic Light' signal for Easy Mode.
    Returns:
        dict with 'signal' (buy/sell/neutral), 'score' (-3 to +3), 'reasons' (list), 'color' (green/red/yellow)
    """
    if df.empty or len(df) < 200:
        return {"signal": "NEUTRAL", "score": 0, "reasons": ["Insufficient data"], "color": "yellow"}
    
    # Ensure we work on a copy
    df = df.copy()
    
    # Calculate indicators if missing
    if "SMA20" not in df.columns:
        df["SMA20"] = _sma(df["close"], 20)
    if "SMA200" not in df.columns:
        df["SMA200"] = _sma(df["close"], 200)
    if "RSI" not in df.columns:
        df["RSI"] = _rsi(df["close"], 14)
    if "MACD" not in df.columns or "MACD Signal" not in df.columns:
        m, s, h = _macd(df["close"])
        df["MACD"] = m
        df["MACD Signal"] = s
    
    last_close = df["close"].iloc[-1]
    sma20 = df["SMA20"].iloc[-1]
    sma200 = df["SMA200"].iloc[-1]
    rsi = df["RSI"].iloc[-1]
    macd = df["MACD"].iloc[-1]
    signal = df["MACD Signal"].iloc[-1]
    
    score = 0
    reasons = []
    
    # 1. Trend (SMA20 vs SMA200)
    if sma20 > sma200:
        score += 1
        reasons.append("Short-term trend is ABOVE long-term trend (Bullish)")
    else:
        score -= 1
        reasons.append("Short-term trend is BELOW long-term trend (Bearish)")
        
    # 2. Momentum (RSI)
    if rsi < 30:
        score += 1
        reasons.append("Price is Oversold (Potential Bounce)")
    elif rsi > 70:
        score -= 1
        reasons.append("Price is Overbought (Potential Drop)")
    else:
        reasons.append("Momentum is Neutral")
        
    # 3. MACD
    if macd > signal:
        score += 1
        reasons.append("MACD Momentum is Positive")
    else:
        score -= 1
        reasons.append("MACD Momentum is Negative")
        
    # Determine final signal
    signal_type = "NEUTRAL"
    color = "#ffcc00"
    
    if score >= 2:
        signal_type = "STRONG BUY"
        color = "#00ff88"
    elif score == 1:
        signal_type = "BUY"
        color = "#00cc66"
    elif score == 0:
        signal_type = "NEUTRAL"
        color = "#ffcc00"
    elif score == -1:
        signal_type = "SELL"
        color = "#ff6666"
    else: # <= -2
        signal_type = "STRONG SELL"
        color = "#ff0000"
        
    # Calculate Trade Suggestion (ATR-based)
    # Ensure ATR is calculated
    if "ATR" not in df.columns:
        atr_series = _atr(df, 14)
        atr = atr_series.iloc[-1] if not atr_series.empty else 0
    else:
        atr = df["ATR"].iloc[-1]
        
    # Fallback if ATR is 0 or nan
    if pd.isna(atr) or atr == 0:
        atr = last_close * 0.02 # Default to 2%
    
    trade_suggestion = {
        "side": "NEUTRAL",
        "entry": last_close,
        "stop_loss": 0.0,
        "take_profit": 0.0,
        "risk_reward": 0.0
    }
    
    if score >= 1: # Bullish
        stop_loss = last_close - (2.0 * atr)
        take_profit = last_close + (3.0 * atr) # 1.5 R/R
        trade_suggestion.update({
            "side": "BUY",
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "risk_reward": 1.5
        })
    elif score <= -1: # Bearish
        stop_loss = last_close + (2.0 * atr)
        take_profit = last_close - (3.0 * atr) # 1.5 R/R
        trade_suggestion.update({
            "side": "SELL",
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "risk_reward": 1.5
        })

    return {
        "signal": signal_type,
        "score": score,
        "reasons": reasons,
        "color": color,
        "trade_suggestion": trade_suggestion
    }

def generate_share_card_html(ticker, price, change_pct, signal_data):
    """Generate HTML for a shareable card."""
    color = signal_data["color"]
    signal_text = signal_data["signal"]
    
    html = f"""
    <div style="
        width: 400px;
        padding: 20px;
        background: linear-gradient(135deg, #0a0e27 0%, #1a1f3a 100%);
        border: 2px solid {color};
        border-radius: 15px;
        font-family: sans-serif;
        color: white;
        text-align: center;
        box-shadow: 0 10px 30px rgba(0,0,0,0.5);
    ">
        <div style="font-size: 24px; font-weight: bold; margin-bottom: 10px; color: #b0b0b0;">TRADERQ ANALYSIS</div>
        <div style="font-size: 48px; font-weight: 800; margin-bottom: 5px; color: #ffffff;">{ticker}</div>
        <div style="font-size: 32px; font-weight: 600; margin-bottom: 20px; color: {'#00ff88' if change_pct >= 0 else '#ff0000'};">
            ${price:,.2f} <span style="font-size: 20px;">({change_pct:+.2f}%)</span>
        </div>
        <div style="
            background-color: {color}20;
            border: 1px solid {color};
            color: {color};
            padding: 10px;
            border-radius: 10px;
            font-size: 28px;
            font-weight: bold;
            margin-bottom: 20px;
        ">
            {signal_text}
        </div>
        <div style="text-align: left; font-size: 14px; color: #cccccc; margin-bottom: 20px;">
            {'<br>• '.join([''] + signal_data['reasons'])}
        </div>
        <div style="font-size: 12px; color: #666666;">Generated by TraderQ AI</div>
    </div>
    """
    return html

def send_email_alert(to_email: str, subject: str, body: str, config: dict) -> bool:
    """Send email alert using SMTP."""
    try:
        import smtplib
        from email.mime.text import MIMEText
        from email.mime.multipart import MIMEMultipart
        
        smtp_server = config.get("smtp_server", "smtp.gmail.com")
        smtp_port = config.get("smtp_port", 587)
        from_email = config.get("from_email", "")
        password = config.get("password", "")
        
        if not from_email or not password:
            return False
        
        msg = MIMEMultipart()
        msg["From"] = from_email
        msg["To"] = to_email
        msg["Subject"] = subject
        msg.attach(MIMEText(body, "plain"))
        
        server = smtplib.SMTP(smtp_server, smtp_port)
        server.starttls()
        server.login(from_email, password)
        server.send_message(msg)
        server.quit()
        return True
    except Exception as e:
        print(f"Email send error: {e}")
        return False

def load_alert_history() -> list:
    """Load alert history from Firestore."""
    try:
        if user_id is None or db is None:
            return []
        return db.get_alert_history(user_id, limit=1000)
    except Exception:
        return []

def save_alert_history(history: list):
    """Save alert history to Firestore."""
    try:
        if user_id is None or db is None:
            st.error("Not authenticated. Please log in.")
            return
        # Clear existing history and add all items
        # Note: For better performance, consider using add_alert_history for individual items
        for item in history:
            db.add_alert_history(user_id, item)
    except Exception as e:
        st.error(f"Failed to save alert history: {e}")

# --- Trade Journal ---
def load_trade_journal() -> list:
    """Load trade journal from Firestore."""
    try:
        if user_id is None or db is None:
            return []
        return db.get_trade_journal(user_id)
    except Exception:
        return []

def save_trade_journal(journal: list):
    """Save trade journal to Firestore."""
    try:
        if user_id is None or db is None:
            st.error("Not authenticated. Please log in.")
            return
        db.save_trade_journal(user_id, journal)
    except Exception as e:
        st.error(f"Failed to save trade journal: {e}")

# --- Fair Value Gap Detection ---
def find_fair_value_gaps(df: pd.DataFrame) -> list:
    """
    Detect Fair Value Gaps (FVG) in price action.
    FVG is a 3-candle pattern where:
    - Candle 1 and Candle 3 overlap
    - Candle 2 creates a gap (doesn't overlap with Candle 1 or 3)
    
    Returns list of FVG dictionaries with start/end price and date range.
    """
    if df.empty or len(df) < 3:
        return []
    
    if "high" not in df.columns or "low" not in df.columns:
        return []
    
    fvgs = []
    
    for i in range(len(df) - 2):
        c1_high = df["high"].iloc[i]
        c1_low = df["low"].iloc[i]
        c2_high = df["high"].iloc[i + 1]
        c2_low = df["low"].iloc[i + 1]
        c3_high = df["high"].iloc[i + 2]
        c3_low = df["low"].iloc[i + 2]
        
        # Bullish FVG: Gap up (candle 2 is above candle 1, and candle 3 overlaps with candle 1)
        # Conditions:
        # 1. Candle 2 low > Candle 1 high (gap up from candle 1)
        # 2. Candle 3 low <= Candle 1 high (candle 3 overlaps with candle 1)
        # 3. Candle 3 high >= Candle 1 low (candle 3 overlaps with candle 1 from below)
        # This creates an unfilled gap between c1_high and c2_low
        if (c2_low > c1_high and c3_low <= c1_high and c3_high >= c1_low):
            fvgs.append({
                "type": "bullish",
                "start_price": c1_high,  # Gap starts here
                "end_price": c2_low,     # Gap ends here
                "start_date": df.index[i],
                "end_date": df.index[i + 2],
                "gap_high": c2_low,      # Top of the gap (candle 2 low)
                "gap_low": c1_high       # Bottom of the gap (candle 1 high)
            })
        
        # Bearish FVG: Gap down (candle 2 is below candle 1, and candle 3 overlaps with candle 1)
        # Conditions:
        # 1. Candle 2 high < Candle 1 low (gap down from candle 1)
        # 2. Candle 3 high >= Candle 1 low (candle 3 overlaps with candle 1)
        # 3. Candle 3 low <= Candle 1 high (candle 3 overlaps with candle 1 from above)
        # This creates an unfilled gap between c2_high and c1_low
        elif (c2_high < c1_low and c3_high >= c1_low and c3_low <= c1_high):
            fvgs.append({
                "type": "bearish",
                "start_price": c1_low,   # Gap starts here
                "end_price": c2_high,    # Gap ends here
                "start_date": df.index[i],
                "end_date": df.index[i + 2],
                "gap_high": c1_low,      # Top of the gap (candle 1 low)
                "gap_low": c2_high       # Bottom of the gap (candle 2 high)
            })
    
    return fvgs

# --- Support & Resistance Detection ---
def find_support_resistance(df: pd.DataFrame, window: int = 20, lookback: int = 5) -> dict:
    """Find support and resistance levels using pivot points."""
    if df.empty or "high" not in df.columns or "low" not in df.columns:
        return {"support": [], "resistance": []}
    
    highs = df["high"].values
    lows = df["low"].values
    
    # Find local maxima (resistance) and minima (support)
    from scipy.signal import argrelextrema
    
    resistance_levels = []
    support_levels = []
    
    # Find resistance (local highs)
    max_indices = argrelextrema(highs, np.greater, order=lookback)[0]
    for idx in max_indices:
        if idx < len(df):
            resistance_levels.append({
                "price": float(highs[idx]),
                "date": df.index[idx],
                "strength": 1  # Can be enhanced with volume/retest count
            })
    
    # Find support (local lows)
    min_indices = argrelextrema(lows, np.less, order=lookback)[0]
    for idx in min_indices:
        if idx < len(df):
            support_levels.append({
                "price": float(lows[idx]),
                "date": df.index[idx],
                "strength": 1
            })
    
    # Sort and get most recent/relevant levels
    resistance_levels.sort(key=lambda x: x["price"], reverse=True)
    support_levels.sort(key=lambda x: x["price"], reverse=True)
    
    # Filter to most significant levels (within 2% of current price)
    current_price = float(df["close"].iloc[-1])
    resistance_levels = [r for r in resistance_levels if abs(r["price"] - current_price) / current_price < 0.2]
    support_levels = [s for s in support_levels if abs(s["price"] - current_price) / current_price < 0.2]
    
    return {
        "support": support_levels[:5],  # Top 5 support levels
        "resistance": resistance_levels[:5]  # Top 5 resistance levels
    }

# --- Pattern Detection ---
def detect_patterns(df: pd.DataFrame) -> list:
    """Detect common chart patterns using local extrema."""
    if df.empty or len(df) < 30:
        return []
    
    patterns = []
    prices = df["close"].values
    highs = df["high"].values
    lows = df["low"].values
    
    # Use scipy to find local peaks and troughs
    from scipy.signal import argrelextrema
    
    # Find local peaks (maxima) and troughs (minima)
    # Look for peaks/troughs within a window of 5-10 periods
    window = min(10, len(highs) // 4)
    if window < 3:
        window = 3
    
    peak_indices = argrelextrema(highs, np.greater, order=window)[0]
    trough_indices = argrelextrema(lows, np.less, order=window)[0]
    
    # Focus on recent data (last 60 periods or all if less)
    lookback = min(60, len(df))
    recent_start = len(df) - lookback
    peak_indices = peak_indices[peak_indices >= recent_start]
    trough_indices = trough_indices[trough_indices >= recent_start]
    
    # Double Top: Two similar peaks with a trough between them
    if len(peak_indices) >= 2:
        for i in range(len(peak_indices) - 1):
            peak1_idx = peak_indices[i]
            peak2_idx = peak_indices[i + 1]
            peak1_val = highs[peak1_idx]
            peak2_val = highs[peak2_idx]
            
            # Check if peaks are similar (within 3%)
            if abs(peak1_val - peak2_val) / max(peak1_val, peak2_val) < 0.03:
                # Check if there's a trough between them
                trough_between = trough_indices[(trough_indices > peak1_idx) & (trough_indices < peak2_idx)]
                if len(trough_between) > 0:
                    # Check if price declined after second peak (bearish)
                    if peak2_idx < len(highs) - 5:
                        post_peak_avg = highs[peak2_idx + 1:peak2_idx + 5].mean()
                        if post_peak_avg < peak2_val * 0.98:  # Price declined after peak
                            patterns.append({
                                "type": "Double Top",
                                "confidence": 0.7,
                                "description": f"Bearish reversal pattern: Two similar peaks at ${peak1_val:.2f} and ${peak2_val:.2f}"
                            })
                            break  # Only report one double top
    
    # Double Bottom: Two similar troughs with a peak between them
    if len(trough_indices) >= 2:
        for i in range(len(trough_indices) - 1):
            trough1_idx = trough_indices[i]
            trough2_idx = trough_indices[i + 1]
            trough1_val = lows[trough1_idx]
            trough2_val = lows[trough2_idx]
            
            # Check if troughs are similar (within 3%)
            if abs(trough1_val - trough2_val) / max(trough1_val, trough2_val) < 0.03:
                # Check if there's a peak between them
                peak_between = peak_indices[(peak_indices > trough1_idx) & (peak_indices < trough2_idx)]
                if len(peak_between) > 0:
                    # Check if price rose after second trough (bullish)
                    if trough2_idx < len(lows) - 5:
                        post_trough_avg = lows[trough2_idx + 1:trough2_idx + 5].mean()
                        if post_trough_avg > trough2_val * 1.02:  # Price rose after trough
                            patterns.append({
                                "type": "Double Bottom",
                                "confidence": 0.7,
                                "description": f"Bullish reversal pattern: Two similar troughs at ${trough1_val:.2f} and ${trough2_val:.2f}"
                            })
                            break  # Only report one double bottom
    
    # Head and Shoulders: Three peaks, middle one highest
    if len(peak_indices) >= 3:
        for i in range(len(peak_indices) - 2):
            left_shoulder_idx = peak_indices[i]
            head_idx = peak_indices[i + 1]
            right_shoulder_idx = peak_indices[i + 2]
            
            left_shoulder = highs[left_shoulder_idx]
            head = highs[head_idx]
            right_shoulder = highs[right_shoulder_idx]
            
            # Head should be highest, shoulders similar
            if (head > left_shoulder * 1.02 and head > right_shoulder * 1.02 and
                abs(left_shoulder - right_shoulder) / max(left_shoulder, right_shoulder) < 0.05):
                # Check for neckline (troughs between peaks)
                trough1 = trough_indices[(trough_indices > left_shoulder_idx) & (trough_indices < head_idx)]
                trough2 = trough_indices[(trough_indices > head_idx) & (trough_indices < right_shoulder_idx)]
                if len(trough1) > 0 and len(trough2) > 0:
                    neckline1 = lows[trough1[0]]
                    neckline2 = lows[trough2[0]]
                    # Neckline should be similar
                    if abs(neckline1 - neckline2) / max(neckline1, neckline2) < 0.05:
                        patterns.append({
                            "type": "Head and Shoulders",
                            "confidence": 0.75,
                            "description": f"Bearish reversal: Head at ${head:.2f}, shoulders at ${left_shoulder:.2f} and ${right_shoulder:.2f}"
                        })
                        break
    
    # Inverse Head and Shoulders: Three troughs, middle one lowest
    if len(trough_indices) >= 3:
        for i in range(len(trough_indices) - 2):
            left_shoulder_idx = trough_indices[i]
            head_idx = trough_indices[i + 1]
            right_shoulder_idx = trough_indices[i + 2]
            
            left_shoulder = lows[left_shoulder_idx]
            head = lows[head_idx]
            right_shoulder = lows[right_shoulder_idx]
            
            # Head should be lowest, shoulders similar
            if (head < left_shoulder * 0.98 and head < right_shoulder * 0.98 and
                abs(left_shoulder - right_shoulder) / max(left_shoulder, right_shoulder) < 0.05):
                # Check for neckline (peaks between troughs)
                peak1 = peak_indices[(peak_indices > left_shoulder_idx) & (peak_indices < head_idx)]
                peak2 = peak_indices[(peak_indices > head_idx) & (peak_indices < right_shoulder_idx)]
                if len(peak1) > 0 and len(peak2) > 0:
                    neckline1 = highs[peak1[0]]
                    neckline2 = highs[peak2[0]]
                    # Neckline should be similar
                    if abs(neckline1 - neckline2) / max(neckline1, neckline2) < 0.05:
                        patterns.append({
                            "type": "Inverse Head and Shoulders",
                            "confidence": 0.75,
                            "description": f"Bullish reversal: Head at ${head:.2f}, shoulders at ${left_shoulder:.2f} and ${right_shoulder:.2f}"
                        })
                        break
    
    # Ascending Triangle: Higher lows, similar highs
    if len(peak_indices) >= 2 and len(trough_indices) >= 2:
        recent_peaks = peak_indices[-3:] if len(peak_indices) >= 3 else peak_indices
        recent_troughs = trough_indices[-3:] if len(trough_indices) >= 3 else trough_indices
        
        if len(recent_peaks) >= 2 and len(recent_troughs) >= 2:
            peak_vals = highs[recent_peaks]
            trough_vals = lows[recent_troughs]
            
            # Peaks should be similar (resistance level)
            peak_std = np.std(peak_vals) / np.mean(peak_vals)
            # Troughs should be ascending
            trough_trend = (trough_vals[-1] - trough_vals[0]) / trough_vals[0]
            
            if peak_std < 0.02 and trough_trend > 0.01:  # Stable resistance, rising support
                patterns.append({
                    "type": "Ascending Triangle",
                    "confidence": 0.65,
                    "description": "Bullish continuation: Rising support meeting resistance"
                })
    
    # Descending Triangle: Lower highs, similar lows
    if len(peak_indices) >= 2 and len(trough_indices) >= 2:
        recent_peaks = peak_indices[-3:] if len(peak_indices) >= 3 else peak_indices
        recent_troughs = trough_indices[-3:] if len(trough_indices) >= 3 else trough_indices
        
        if len(recent_peaks) >= 2 and len(recent_troughs) >= 2:
            peak_vals = highs[recent_peaks]
            trough_vals = lows[recent_troughs]
            
            # Troughs should be similar (support level)
            trough_std = np.std(trough_vals) / np.mean(trough_vals)
            # Peaks should be descending
            peak_trend = (peak_vals[-1] - peak_vals[0]) / peak_vals[0]
            
            if trough_std < 0.02 and peak_trend < -0.01:  # Stable support, falling resistance
                patterns.append({
                    "type": "Descending Triangle",
                    "confidence": 0.65,
                    "description": "Bearish continuation: Falling resistance meeting support"
                })
    
    # If no patterns found, check for basic trend patterns
    if not patterns and len(df) >= 20:
        recent_prices = prices[-20:]
        price_trend = (recent_prices[-1] - recent_prices[0]) / recent_prices[0]
        
        if price_trend > 0.05:
            patterns.append({
                "type": "Uptrend",
                "confidence": 0.5,
                "description": f"Strong upward trend: {price_trend*100:.1f}% gain in last 20 periods"
            })
        elif price_trend < -0.05:
            patterns.append({
                "type": "Downtrend",
                "confidence": 0.5,
                "description": f"Strong downward trend: {price_trend*100:.1f}% decline in last 20 periods"
            })
        else:
            patterns.append({
                "type": "Sideways/Consolidation",
                "confidence": 0.4,
                "description": "Price moving in a range, no clear trend"
            })
    
    return patterns

# --- Correlation Analysis ---
def calculate_correlation(tickers: list[str], start: date, end: date, mode: str, interval: str = "1d") -> pd.DataFrame:
    """Calculate correlation matrix for multiple tickers."""
    if len(tickers) < 2:
        return pd.DataFrame()
    
    prices = {}
    for ticker in tickers:
        df = load_data(ticker, start, end, mode, interval=interval, data_source="Yahoo Finance")
        if not df.empty and "close" in df.columns:
            prices[ticker] = df["close"]
    
    if len(prices) < 2:
        return pd.DataFrame()
    
    # Align all price series
    price_df = pd.DataFrame(prices)
    price_df = price_df.dropna()
    
    if price_df.empty:
        return pd.DataFrame()
    
    # Calculate returns
    returns = price_df.pct_change().dropna()
    
    # Calculate correlation
    correlation = returns.corr()
    return correlation

# --- Sentiment Analysis ---
def analyze_sentiment(text: str) -> dict:
    """Analyze sentiment of text and return bullish/bearish classification."""
    if not text or len(text.strip()) == 0:
        return {"sentiment": "NEUTRAL", "score": 0.0, "confidence": 0.0}
    
    text_lower = text.lower()
    
    # Bullish keywords
    bullish_keywords = [
        "surge", "rally", "gain", "rise", "up", "bullish", "buy", "strong", "growth",
        "profit", "beat", "exceed", "outperform", "positive", "optimistic", "upgrade",
        "breakthrough", "success", "win", "soar", "jump", "climb", "advance", "boom"
    ]
    
    # Bearish keywords
    bearish_keywords = [
        "drop", "fall", "decline", "down", "bearish", "sell", "weak", "loss", "miss",
        "disappoint", "underperform", "negative", "pessimistic", "downgrade", "crash",
        "plunge", "tumble", "slump", "sink", "retreat", "worry", "concern", "risk",
        "warning", "crisis", "recession", "bankruptcy", "lawsuit", "investigation"
    ]
    
    # Count keyword matches
    bullish_count = sum(1 for word in bullish_keywords if word in text_lower)
    bearish_count = sum(1 for word in bearish_keywords if word in text_lower)
    
    # Calculate sentiment score (-1 to 1)
    total_keywords = bullish_count + bearish_count
    if total_keywords == 0:
        score = 0.0
    else:
        score = (bullish_count - bearish_count) / max(total_keywords, 1)
    
    # Determine sentiment
    if score > 0.2:
        sentiment = "BULLISH"
        confidence = min(abs(score) * 2, 1.0)
    elif score < -0.2:
        sentiment = "BEARISH"
        confidence = min(abs(score) * 2, 1.0)
    else:
        sentiment = "NEUTRAL"
        confidence = 1.0 - abs(score)
    
    return {
        "sentiment": sentiment,
        "score": round(score, 2),
        "confidence": round(confidence, 2),
        "bullish_keywords": bullish_count,
        "bearish_keywords": bearish_count
    }

# --- News Fetching ---
def fetch_news(ticker: str, max_items: int = 5) -> list:
    """Fetch news for a ticker using Yahoo Finance."""
    try:
        yf = _yf()
        stock = yf.Ticker(ticker)
        
        # Try multiple methods to get news
        news = None
        
        # Method 1: Direct property access
        if hasattr(stock, 'news'):
            try:
                news = stock.news
                if news and len(news) > 0:
                    pass  # Got news, continue
                else:
                    news = None
            except Exception as e:
                import sys
                print(f"Method 1 (news property) failed: {e}", file=sys.stderr)
                news = None
        
        # Method 2: Try get_news() if it exists
        if (news is None or (isinstance(news, list) and len(news) == 0)) and hasattr(stock, 'get_news'):
            try:
                news = stock.get_news()
            except Exception as e:
                import sys
                print(f"Method 2 (get_news) failed: {e}", file=sys.stderr)
        
        # Method 3: Try accessing via info or other attributes
        if (news is None or (isinstance(news, list) and len(news) == 0)) and hasattr(stock, 'info'):
            try:
                info = stock.info
                if isinstance(info, dict) and 'news' in info:
                    news = info['news']
            except Exception:
                pass
        
        # Debug: Print what we got
        import sys
        if news is None:
            print(f"No news found for {ticker} - news is None", file=sys.stderr)
        elif isinstance(news, list) and len(news) == 0:
            print(f"Empty news list for {ticker}", file=sys.stderr)
        else:
            print(f"Found news for {ticker}: type={type(news)}, length={len(news) if isinstance(news, list) else 'N/A'}", file=sys.stderr)
            if isinstance(news, list) and len(news) > 0:
                print(f"First item type: {type(news[0])}", file=sys.stderr)
                if isinstance(news[0], dict):
                    print(f"First item keys: {list(news[0].keys())}", file=sys.stderr)
                    # Print first item for debugging
                    print(f"First item sample: {str(news[0])[:200]}", file=sys.stderr)
        
        # Handle different return types
        if news is None:
            return []
        
        # Check if it's an empty list
        if isinstance(news, list) and len(news) == 0:
            return []
        
        # If it's a list, process it
        if isinstance(news, list):
            if len(news) > 0:
                # Add sentiment analysis to each article
                enriched_news = []
                for item in news[:max_items]:
                    # Normalize the item - yfinance news items can have different structures
                    normalized_item = {}
                    
                    # Handle both dict and object types
                    if not isinstance(item, dict):
                        # Convert to dict if it's an object
                        try:
                            if hasattr(item, '__dict__'):
                                item = vars(item)
                            elif hasattr(item, 'keys'):
                                item = dict(item)
                            else:
                                item = {}
                        except Exception:
                            item = {}
                    
                    # NEW: Handle the case where content is a string containing dict data
                    content_data = {}
                    if 'content' in item:
                        content_str = item.get('content', '')
                        if isinstance(content_str, str):
                            # Try to parse the content string as JSON or eval it
                            try:
                                import ast
                                # Try to safely evaluate the string as a dict
                                content_data = ast.literal_eval(content_str)
                            except Exception:
                                try:
                                    import json
                                    content_data = json.loads(content_str.replace("'", '"'))
                                except Exception:
                                    # If parsing fails, use the string as-is
                                    content_data = {'rawContent': content_str}
                        elif isinstance(content_str, dict):
                            content_data = content_str
                    
                    # Merge content_data into item for easier access
                    if content_data:
                        item = {**item, **content_data}
                    
                    # Try to extract title from various possible fields (check more variations)
                    normalized_item['title'] = (
                        item.get('title') or 
                        item.get('headline') or 
                        item.get('headlineText') or
                        item.get('headlineTextFull') or
                        item.get('titleText') or
                        item.get('longTitle') or
                        str(item.get('uuid', '')) if item.get('uuid') else
                        str(item.get('id', '')) if item.get('id') else
                        f"Article {len(enriched_news) + 1}"
                    )
                    
                    # Try to extract publisher/source (check more variations)
                    normalized_item['publisher'] = (
                        item.get('publisher') or 
                        item.get('source') or 
                        item.get('provider') or
                        item.get('providerName') or
                        item.get('publisherName') or
                        item.get('publisherDisplayName') or
                        item.get('providerDisplayName') or
                        "Unknown"
                    )
                    
                    # Try to extract summary/description - check ALL possible fields
                    normalized_item['summary'] = (
                        item.get('summary') or 
                        item.get('description') or 
                        item.get('text') or
                        item.get('body') or
                        item.get('excerpt') or
                        item.get('snippet') or
                        item.get('preview') or
                        item.get('rawContent') or
                        ""
                    )
                    
                    # Try to extract link - check multiple URL fields
                    normalized_item['link'] = (
                        item.get('link') or 
                        item.get('url') or
                        item.get('canonicalUrl') or
                        item.get('canonical_url') or
                        item.get('webUrl') or
                        item.get('web_url') or
                        item.get('clickThroughUrl') or
                        item.get('clickThroughUrl.raw') or
                        ""
                    )
                    
                    # Try to extract publish time
                    normalized_item['providerPublishTime'] = (
                        item.get('providerPublishTime') or
                        item.get('pubDate') or
                        item.get('publishedAt') or
                        item.get('time') or
                        item.get('publishTime') or
                        item.get('timestamp') or
                        item.get('created') or
                        item.get('publishTime.raw') or
                        None
                    )
                    
                    # Store the raw item for debugging
                    normalized_item['_raw'] = item
                    
                    # Combine title and summary for sentiment analysis
                    text_for_sentiment = normalized_item['title']
                    if normalized_item['summary']:
                        text_for_sentiment += " " + normalized_item['summary']
                    
                    # Only analyze if we have some text
                    if text_for_sentiment.strip():
                        sentiment = analyze_sentiment(text_for_sentiment)
                        normalized_item['sentiment'] = sentiment
                    else:
                        normalized_item['sentiment'] = {"sentiment": "NEUTRAL", "score": 0.0, "confidence": 0.0}
                    
                    enriched_news.append(normalized_item)
                
                return enriched_news
            return []
        
        # If it's a dict with a 'news' key
        if isinstance(news, dict) and 'news' in news:
            news_list = news['news']
            if isinstance(news_list, list) and len(news_list) > 0:
                enriched_news = []
                for item in news_list[:max_items]:
                    normalized_item = {}
                    normalized_item['title'] = item.get('title') or item.get('headline') or "Article"
                    normalized_item['publisher'] = item.get('publisher') or item.get('source') or "Unknown"
                    normalized_item['summary'] = item.get('summary') or item.get('description') or ""
                    normalized_item['link'] = item.get('link') or item.get('url') or ""
                    normalized_item['providerPublishTime'] = item.get('providerPublishTime') or item.get('pubDate')
                    
                    text_for_sentiment = normalized_item['title']
                    if normalized_item['summary']:
                        text_for_sentiment += " " + normalized_item['summary']
                    
                    if text_for_sentiment.strip():
                        sentiment = analyze_sentiment(text_for_sentiment)
                        normalized_item['sentiment'] = sentiment
                    else:
                        normalized_item['sentiment'] = {"sentiment": "NEUTRAL", "score": 0.0, "confidence": 0.0}
                    
                    enriched_news.append(normalized_item)
                
                return enriched_news
        
        return []
    except Exception as e:
        # Log the error for debugging but don't crash
        import sys
        print(f"Error fetching news for {ticker}: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return []

# --- AI Recommendation System ---
def generate_ai_recommendation(ticker: str, df: pd.DataFrame, start: date, end: date, mode: str, interval: str,
                               use_sma: bool = True, use_rsi: bool = True, use_macd: bool = True,
                               use_supertrend: bool = True, use_fvg: bool = True, use_news: bool = True,
                               macd_mode: str = "Extended",
                               macd_sideways_window: int = 10,
                               macd_sideways_threshold: float = 8.0) -> dict:
    """Generate comprehensive AI trading recommendation based on selected indicators."""
    if df.empty or len(df) < 200:
        return {"error": "Insufficient data for analysis"}
    
    d = df.copy()
    d["SMA20"] = _sma(d["close"], SMA_SHORT)
    d["SMA200"] = _sma(d["close"], SMA_LONG)
    
    last = d.iloc[-1]
    prev = d.iloc[-2] if len(d) > 1 else last
    
    # Current price
    current_price = float(last["close"])
    sma20_val = float(last["SMA20"])
    sma200_val = float(last["SMA200"])
    
    # Calculate RSI
    rsi = _rsi(d["close"], window=14)
    rsi_val = float(rsi.iloc[-1]) if not pd.isna(rsi.iloc[-1]) else 50
    
    # Calculate MACD according to selected mode
    if macd_mode == "Normal":
        macd_line, signal_line, macd_hist = _macd(d["close"], fast=12, slow=26, signal=9)
    else:
        # Extended MACD (flattens during sideways markets)
            macd_line, signal_line, macd_hist = _macd_extended(
                d["close"], fast=12, slow=26, signal=9,
                sideways_window=macd_sideways_window,
                sideways_threshold=macd_sideways_threshold / 100.0
            )
    macd_val = float(macd_line.iloc[-1]) if not pd.isna(macd_line.iloc[-1]) else 0
    signal_val = float(signal_line.iloc[-1]) if not pd.isna(signal_line.iloc[-1]) else 0
    macd_hist_val = float(macd_hist.iloc[-1]) if not pd.isna(macd_hist.iloc[-1]) else 0
    macd_prev = float(macd_line.iloc[-2]) if len(macd_line) > 1 and not pd.isna(macd_line.iloc[-2]) else macd_val
    signal_prev = float(signal_line.iloc[-2]) if len(signal_line) > 1 and not pd.isna(signal_line.iloc[-2]) else signal_val
    
    # Calculate Supertrend
    supertrend, trend = _supertrend(d, period=10, multiplier=3.0)
    trend_current = float(trend.iloc[-1]) if not pd.isna(trend.iloc[-1]) else 0
    trend_prev = float(trend.iloc[-2]) if len(trend) > 1 and not pd.isna(trend.iloc[-2]) else trend_current
    supertrend_val = float(supertrend.iloc[-1]) if not pd.isna(supertrend.iloc[-1]) else current_price
    
    # Detect Fair Value Gaps
    fvgs = find_fair_value_gaps(d)
    # Get FVGs from last 10 candles
    if len(d) >= 10:
        recent_cutoff = d.index[-10]
        recent_fvgs = [fvg for fvg in fvgs if fvg["end_date"] >= recent_cutoff]
    else:
        recent_fvgs = fvgs
    
    # Analyze Supertrend (only if enabled)
    supertrend_analysis = {}
    supertrend_score = 0
    if not use_supertrend:
        supertrend_analysis["status"] = "Disabled"
        supertrend_analysis["description"] = "Supertrend analysis is disabled"
        supertrend_score = 0
    elif trend_current > 0:
        # Uptrend
        supertrend_analysis["status"] = "Bullish"
        supertrend_analysis["description"] = f"Supertrend indicates UPTREND - Price is above Supertrend line (${supertrend_val:.2f})"
        supertrend_score = 2
        if trend_prev <= 0:
            # Trend changed from down to up - strong signal
            supertrend_analysis["status"] = "Bullish Reversal"
            supertrend_analysis["description"] = "🟢 BULLISH REVERSAL! Supertrend just turned bullish - Strong buy signal"
            supertrend_score = 3
    elif trend_current < 0:
        # Downtrend
        supertrend_analysis["status"] = "Bearish"
        supertrend_analysis["description"] = f"Supertrend indicates DOWNTREND - Price is below Supertrend line (${supertrend_val:.2f})"
        supertrend_score = -2
        if trend_prev >= 0:
            # Trend changed from up to down - strong signal
            supertrend_analysis["status"] = "Bearish Reversal"
            supertrend_analysis["description"] = "🔴 BEARISH REVERSAL! Supertrend just turned bearish - Strong sell signal"
            supertrend_score = -3
    else:
        supertrend_analysis["status"] = "Neutral"
        supertrend_analysis["description"] = f"Supertrend is neutral (${supertrend_val:.2f})"
        supertrend_score = 0
    
    # Analyze SMA Cross (only if enabled)
    sma_analysis = {}
    sma_score = 0
    if not use_sma:
        sma_analysis["status"] = "Disabled"
        sma_analysis["description"] = "SMA analysis is disabled"
        sma_score = 0
    elif sma20_val > sma200_val:
        sma_analysis["status"] = "Bullish"
        sma_analysis["description"] = f"SMA20 (${sma20_val:.2f}) is above SMA200 (${sma200_val:.2f})"
        sma_score = 2
        if prev["SMA20"] <= prev["SMA200"]:
            sma_analysis["status"] = "Golden Cross"
            sma_analysis["description"] = "🟢 GOLDEN CROSS detected! SMA20 just crossed above SMA200 - Strong bullish signal"
            sma_score = 3
    elif sma20_val < sma200_val:
        sma_analysis["status"] = "Bearish"
        sma_analysis["description"] = f"SMA20 (${sma20_val:.2f}) is below SMA200 (${sma200_val:.2f})"
        sma_score = -2
        if prev["SMA20"] >= prev["SMA200"]:
            sma_analysis["status"] = "Death Cross"
            sma_analysis["description"] = "🔴 DEATH CROSS detected! SMA20 just crossed below SMA200 - Strong bearish signal"
            sma_score = -3
    else:
        sma_analysis["status"] = "Neutral"
        sma_analysis["description"] = "SMA20 and SMA200 are at similar levels"
        sma_score = 0
    
    # Analyze RSI (only if enabled)
    rsi_analysis = {}
    rsi_score = 0
    if not use_rsi:
        rsi_analysis["status"] = "Disabled"
        rsi_analysis["description"] = "RSI analysis is disabled"
        rsi_score = 0
    elif rsi_val < 30:
        rsi_analysis["status"] = "Oversold"
        rsi_analysis["description"] = f"RSI at {rsi_val:.1f} - Oversold condition, potential buying opportunity"
        rsi_score = 2
    elif rsi_val > 70:
        rsi_analysis["status"] = "Overbought"
        rsi_analysis["description"] = f"RSI at {rsi_val:.1f} - Overbought condition, potential selling opportunity"
        rsi_score = -2
    elif 30 <= rsi_val <= 50:
        rsi_analysis["status"] = "Neutral-Bullish"
        rsi_analysis["description"] = f"RSI at {rsi_val:.1f} - Neutral to slightly bullish"
        rsi_score = 1
    elif 50 < rsi_val <= 70:
        rsi_analysis["status"] = "Neutral-Bearish"
        rsi_analysis["description"] = f"RSI at {rsi_val:.1f} - Neutral to slightly bearish"
        rsi_score = -1
    else:
        rsi_analysis["status"] = "Neutral"
        rsi_analysis["description"] = f"RSI at {rsi_val:.1f} - Neutral"
        rsi_score = 0
    
    # Analyze MACD (only if enabled)
    macd_analysis = {}
    macd_score = 0
    if not use_macd:
        macd_analysis["status"] = "Disabled"
        macd_analysis["description"] = "MACD analysis is disabled"
        macd_score = 0
    else:
        # Detect if the recent market is sideways; if so, treat MACD as neutral/flat
        sideways_series = _sideways_mask(
            d["close"],
            window=macd_sideways_window,
            threshold=macd_sideways_threshold / 100.0
        )
        is_sideways = bool(sideways_series.iloc[-1]) if len(sideways_series) > 0 else False

        if is_sideways:
            macd_analysis["status"] = "Sideways / Flat"
            macd_analysis["description"] = "Market is range-bound; Extended MACD is flattened to reflect sideways conditions (neutral signal)"
            macd_score = 0
        else:
            # Check for MACD crossovers
            macd_above_signal = macd_val > signal_val
            macd_above_signal_prev = macd_prev > signal_prev
            macd_cross_up = not macd_above_signal_prev and macd_above_signal
            macd_cross_dn = macd_above_signal_prev and not macd_above_signal
            
            if macd_cross_up:
                macd_analysis["status"] = "Bullish Crossover"
                macd_analysis["description"] = f"🟢 MACD BULLISH CROSSOVER! MACD ({macd_val:.4f}) just crossed above Signal ({signal_val:.4f}) - Strong buy signal"
                macd_score = 3
            elif macd_cross_dn:
                macd_analysis["status"] = "Bearish Crossover"
                macd_analysis["description"] = f"🔴 MACD BEARISH CROSSOVER! MACD ({macd_val:.4f}) just crossed below Signal ({signal_val:.4f}) - Strong sell signal"
                macd_score = -3
            elif macd_above_signal:
                if macd_hist_val > 0:
                    macd_analysis["status"] = "Bullish"
                    macd_analysis["description"] = f"MACD ({macd_val:.4f}) above Signal ({signal_val:.4f}) with positive histogram - Bullish momentum"
                    macd_score = 2
                else:
                    macd_analysis["status"] = "Bullish (Weakening)"
                    macd_analysis["description"] = f"MACD ({macd_val:.4f}) above Signal ({signal_val:.4f}) but histogram negative - Bullish momentum weakening"
                    macd_score = 1
            else:
                if macd_hist_val < 0:
                    macd_analysis["status"] = "Bearish"
                    macd_analysis["description"] = f"MACD ({macd_val:.4f}) below Signal ({signal_val:.4f}) with negative histogram - Bearish momentum"
                    macd_score = -2
                else:
                    macd_analysis["status"] = "Bearish (Weakening)"
                    macd_analysis["description"] = f"MACD ({macd_val:.4f}) below Signal ({signal_val:.4f}) but histogram positive - Bearish momentum weakening"
                    macd_score = -1
    
    # Analyze Fair Value Gaps (only if enabled)
    fvg_analysis = {}
    fvg_score = 0
    if not use_fvg:
        fvg_analysis["status"] = "Disabled"
        fvg_analysis["description"] = "Fair Value Gap analysis is disabled"
        fvg_score = 0
    else:
        bullish_fvgs = [fvg for fvg in recent_fvgs if fvg["type"] == "bullish"]
        bearish_fvgs = [fvg for fvg in recent_fvgs if fvg["type"] == "bearish"]
        
        # Check if price is near or in a FVG zone (potential support/resistance)
        fvg_signals = []
        for fvg in recent_fvgs[-3:]:  # Check last 3 FVGs
            gap_high = fvg["gap_high"]
            gap_low = fvg["gap_low"]
            if gap_low <= current_price <= gap_high:
                # Price is in the FVG zone
                if fvg["type"] == "bullish":
                    fvg_signals.append(f"Price is in BULLISH FVG zone (${gap_low:.2f}-${gap_high:.2f}) - Potential support")
                    fvg_score += 1
                else:
                    fvg_signals.append(f"Price is in BEARISH FVG zone (${gap_low:.2f}-${gap_high:.2f}) - Potential resistance")
                    fvg_score -= 1
            elif current_price > gap_high and fvg["type"] == "bullish":
                # Price broke above bullish FVG - bullish signal
                fvg_signals.append(f"Price broke above BULLISH FVG (${gap_low:.2f}-${gap_high:.2f}) - Bullish continuation")
                fvg_score += 1
            elif current_price < gap_low and fvg["type"] == "bearish":
                # Price broke below bearish FVG - bearish signal
                fvg_signals.append(f"Price broke below BEARISH FVG (${gap_low:.2f}-${gap_high:.2f}) - Bearish continuation")
                fvg_score -= 1
        
        if len(bullish_fvgs) > len(bearish_fvgs) and len(bullish_fvgs) > 0:
            fvg_analysis["status"] = "Bullish FVG Pattern"
            fvg_analysis["description"] = f"Recent bullish FVG activity ({len(bullish_fvgs)} bullish vs {len(bearish_fvgs)} bearish). " + "; ".join(fvg_signals) if fvg_signals else "No current FVG price action"
            if not fvg_signals:
                fvg_score = 1
        elif len(bearish_fvgs) > len(bullish_fvgs) and len(bearish_fvgs) > 0:
            fvg_analysis["status"] = "Bearish FVG Pattern"
            fvg_analysis["description"] = f"Recent bearish FVG activity ({len(bearish_fvgs)} bearish vs {len(bullish_fvgs)} bullish). " + "; ".join(fvg_signals) if fvg_signals else "No current FVG price action"
            if not fvg_signals:
                fvg_score = -1
        else:
            fvg_analysis["status"] = "Neutral"
            fvg_analysis["description"] = "; ".join(fvg_signals) if fvg_signals else f"No recent FVG activity ({len(recent_fvgs)} FVGs detected in last 10 candles)"
            fvg_score = 0
    
    # Fetch and analyze news sentiment (only if enabled)
    news_items = fetch_news(ticker, max_items=10) if use_news else []
    news_analysis = {}
    news_score = 0
    
    if not use_news:
        news_analysis["status"] = "Disabled"
        news_analysis["description"] = "News sentiment analysis is disabled"
        news_score = 0
    elif news_items:
        bullish_count = sum(1 for item in news_items if item.get('sentiment', {}).get('sentiment') == 'BULLISH')
        bearish_count = sum(1 for item in news_items if item.get('sentiment', {}).get('sentiment') == 'BEARISH')
        neutral_count = len(news_items) - bullish_count - bearish_count
        
        total_confidence = sum(item.get('sentiment', {}).get('confidence', 0) for item in news_items) / len(news_items) if news_items else 0
        
        if bullish_count > bearish_count:
            news_analysis["status"] = "Bullish"
            news_analysis["description"] = f"News sentiment: {bullish_count} bullish, {bearish_count} bearish, {neutral_count} neutral articles"
            news_score = 2
        elif bearish_count > bullish_count:
            news_analysis["status"] = "Bearish"
            news_analysis["description"] = f"News sentiment: {bullish_count} bullish, {bearish_count} bearish, {neutral_count} neutral articles"
            news_score = -2
        else:
            news_analysis["status"] = "Neutral"
            news_analysis["description"] = f"News sentiment: {bullish_count} bullish, {bearish_count} bearish, {neutral_count} neutral articles"
            news_score = 0
        
        news_analysis["bullish_count"] = bullish_count
        news_analysis["bearish_count"] = bearish_count
        news_analysis["neutral_count"] = neutral_count
        news_analysis["confidence"] = total_confidence
    else:
        news_analysis["status"] = "No Data"
        news_analysis["description"] = "No news articles found for sentiment analysis"
        news_score = 0
    
    # Combine scores
    total_score = sma_score + rsi_score + macd_score + supertrend_score + fvg_score + news_score
    
    # Determine recommendation
    if total_score >= 5:
        recommendation = "STRONG LONG"
        direction = "LONG"
        confidence = "HIGH"
    elif total_score >= 2:
        recommendation = "LONG"
        direction = "LONG"
        confidence = "MEDIUM"
    elif total_score <= -5:
        recommendation = "STRONG SHORT"
        direction = "SHORT"
        confidence = "HIGH"
    elif total_score <= -2:
        recommendation = "SHORT"
        direction = "SHORT"
        confidence = "MEDIUM"
    else:
        recommendation = "NEUTRAL/HOLD"
        direction = "HOLD"
        confidence = "LOW"
    
    # Calculate entry, stop loss, and target prices
    entry_price = current_price
    
    # Calculate ATR for stop loss
    atr_stop = calculate_atr_stop_loss(d, atr_multiplier=2.0, is_long=(direction == "LONG"))
    
    # Get support/resistance levels
    sr_levels = find_support_resistance(d)
    
    if direction == "LONG":
        # For long positions
        stop_loss = min(atr_stop, current_price * 0.95)  # Use ATR or 5% whichever is closer
        
        # Target based on resistance levels or ATR
        resistance_levels = sr_levels.get("resistance", [])
        if resistance_levels:
            nearest_resistance = min([r["price"] for r in resistance_levels if r["price"] > current_price], default=None)
            if nearest_resistance:
                target_price = nearest_resistance
            else:
                target_price = current_price * 1.10  # 10% target if no resistance
        else:
            target_price = current_price * 1.10  # 10% target
        
        # Ensure minimum 2:1 risk/reward
        risk = abs(entry_price - stop_loss)
        reward = abs(target_price - entry_price)
        if reward / risk < 2.0:
            target_price = entry_price + (risk * 2.0)
    elif direction == "SHORT":
        # For short positions
        stop_loss = max(atr_stop, current_price * 1.05)  # Use ATR or 5% whichever is closer
        
        # Target based on support levels or ATR
        support_levels = sr_levels.get("support", [])
        if support_levels:
            nearest_support = max([s["price"] for s in support_levels if s["price"] < current_price], default=None)
            if nearest_support:
                target_price = nearest_support
            else:
                target_price = current_price * 0.90  # 10% target if no support
        else:
            target_price = current_price * 0.90  # 10% target
        
        # Ensure minimum 2:1 risk/reward
        risk = abs(stop_loss - entry_price)
        reward = abs(entry_price - target_price)
        if reward / risk < 2.0:
            target_price = entry_price - (risk * 2.0)
    else:
        # Neutral/Hold
        stop_loss = current_price * 0.95
        target_price = current_price * 1.05
    
    # Calculate risk/reward ratio
    if direction == "LONG":
        risk = abs(entry_price - stop_loss)
        reward = abs(target_price - entry_price)
    elif direction == "SHORT":
        risk = abs(stop_loss - entry_price)
        reward = abs(entry_price - target_price)
    else:
        risk = 0
        reward = 0
    
    risk_reward_ratio = reward / risk if risk > 0 else 0
    
    return {
        "ticker": ticker,
        "current_price": current_price,
        "recommendation": recommendation,
        "direction": direction,
        "confidence": confidence,
        "total_score": total_score,
        "sma_analysis": sma_analysis,
        "rsi_analysis": rsi_analysis,
        "macd_analysis": macd_analysis,
        "supertrend_analysis": supertrend_analysis,
        "fvg_analysis": fvg_analysis,
        "news_analysis": news_analysis,
        "entry_price": entry_price,
        "stop_loss": stop_loss,
        "target_price": target_price,
        "risk_reward_ratio": risk_reward_ratio,
        "risk_pct": (abs(entry_price - stop_loss) / entry_price * 100) if entry_price > 0 else 0,
        "reward_pct": (abs(target_price - entry_price) / entry_price * 100) if entry_price > 0 else 0
    }

# --- Risk Management ---
def calculate_position_size(account_size: float, risk_percent: float, entry_price: float, stop_loss: float) -> dict:
    """Calculate position size based on risk management."""
    if entry_price <= 0 or stop_loss <= 0 or risk_percent <= 0:
        return {"shares": 0, "dollar_risk": 0, "position_value": 0}
    
    risk_per_share = abs(entry_price - stop_loss)
    if risk_per_share == 0:
        return {"shares": 0, "dollar_risk": 0, "position_value": 0}
    
    dollar_risk = account_size * (risk_percent / 100.0)
    shares = dollar_risk / risk_per_share
    position_value = shares * entry_price
    
    return {
        "shares": round(shares, 2),
        "dollar_risk": round(dollar_risk, 2),
        "position_value": round(position_value, 2),
        "risk_per_share": round(risk_per_share, 2)
    }

def calculate_atr_stop_loss(df: pd.DataFrame, atr_multiplier: float = 2.0, is_long: bool = True) -> float:
    """Calculate ATR-based stop loss."""
    if df.empty or len(df) < 14:
        return 0.0
    
    # Calculate ATR (Average True Range)
    high_low = df["high"] - df["low"]
    high_close = abs(df["high"] - df["close"].shift(1))
    low_close = abs(df["low"] - df["close"].shift(1))
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = true_range.rolling(window=14).mean().iloc[-1]
    
    current_price = float(df["close"].iloc[-1])
    
    if is_long:
        stop_loss = current_price - (atr * atr_multiplier)
    else:
        stop_loss = current_price + (atr * atr_multiplier)
    
    return round(stop_loss, 2)

def calculate_risk_reward(entry_price: float, stop_loss: float, target_price: float) -> dict:
    """Calculate risk/reward ratio."""
    if entry_price <= 0:
        return {"risk_reward_ratio": 0, "risk": 0, "reward": 0, "risk_pct": 0, "reward_pct": 0}
    
    risk = abs(entry_price - stop_loss)
    reward = abs(target_price - entry_price)
    
    risk_pct = (risk / entry_price) * 100
    reward_pct = (reward / entry_price) * 100
    
    risk_reward_ratio = reward / risk if risk > 0 else 0
    
    return {
        "risk_reward_ratio": round(risk_reward_ratio, 2),
        "risk": round(risk, 2),
        "reward": round(reward, 2),
        "risk_pct": round(risk_pct, 2),
        "reward_pct": round(reward_pct, 2)
    }

# --- Signal Generator ---
def generate_trading_signal(df: pd.DataFrame) -> dict:
    """Generate trading signal by combining multiple indicators."""
    if df.empty or len(df) < 200:
        return {"signal": "NEUTRAL", "strength": 0, "details": {}}
    
    d = df.copy()
    d["SMA20"] = _sma(d["close"], SMA_SHORT)
    d["SMA200"] = _sma(d["close"], SMA_LONG)
    
    last = d.iloc[-1]
    prev = d.iloc[-2] if len(d) > 1 else last
    
    # Calculate indicators
    rsi = _rsi(d["close"], window=14)
    rsi_val = float(rsi.iloc[-1]) if not pd.isna(rsi.iloc[-1]) else 50
    
    macd_line, signal_line, histogram = _macd(d["close"])
    macd_val = float(macd_line.iloc[-1]) if not pd.isna(macd_line.iloc[-1]) else 0
    signal_val = float(signal_line.iloc[-1]) if not pd.isna(signal_line.iloc[-1]) else 0
    hist_val = float(histogram.iloc[-1]) if not pd.isna(histogram.iloc[-1]) else 0
    
    sma20_val = float(last["SMA20"])
    sma200_val = float(last["SMA200"])
    price = float(last["close"])
    
    # Signal scoring (0-10)
    score = 0
    signal_type = "NEUTRAL"
    details = {}
    
    # SMA Cross (0-3 points)
    if sma20_val > sma200_val:
        score += 2
        details["sma_trend"] = "Bullish"
        if prev["SMA20"] <= prev["SMA200"]:
            score += 1  # Golden Cross bonus
            details["sma_cross"] = "Golden Cross"
    elif sma20_val < sma200_val:
        score -= 2
        details["sma_trend"] = "Bearish"
        if prev["SMA20"] >= prev["SMA200"]:
            score -= 1  # Death Cross penalty
            details["sma_cross"] = "Death Cross"
    else:
        details["sma_trend"] = "Neutral"
    
    # RSI (0-2 points)
    if 30 < rsi_val < 70:
        score += 1
        details["rsi"] = f"Neutral ({rsi_val:.1f})"
    elif rsi_val < 30:
        score += 2
        details["rsi"] = f"Oversold ({rsi_val:.1f})"
    elif rsi_val > 70:
        score -= 2
        details["rsi"] = f"Overbought ({rsi_val:.1f})"
    
    # MACD (0-2 points)
    if macd_val > signal_val and hist_val > 0:
        score += 2
        details["macd"] = "Bullish"
    elif macd_val < signal_val and hist_val < 0:
        score -= 2
        details["macd"] = "Bearish"
    else:
        details["macd"] = "Neutral"
    
    # Volume confirmation (0-1 point)
    if not d["volume"].isna().all():
        vol_sma = _volume_sma(d["volume"], window=20)
        vol_ratio = float(last["volume"]) / float(vol_sma.iloc[-1]) if not pd.isna(vol_sma.iloc[-1]) and vol_sma.iloc[-1] > 0 else 1
        if vol_ratio > 1.2:
            score += 1
            details["volume"] = f"Above average ({vol_ratio:.2f}x)"
        else:
            details["volume"] = f"Below average ({vol_ratio:.2f}x)"
    
    # Determine signal
    if score >= 6:
        signal_type = "STRONG_BUY"
    elif score >= 3:
        signal_type = "BUY"
    elif score <= -6:
        signal_type = "STRONG_SELL"
    elif score <= -3:
        signal_type = "SELL"
    else:
        signal_type = "NEUTRAL"
    
    # Calculate entry/exit recommendations
    entry_price = price
    stop_loss = calculate_atr_stop_loss(d, atr_multiplier=2.0, is_long=(score > 0))
    
    # Target price based on support/resistance
    sr_levels = find_support_resistance(d)
    if score > 0:  # Buy signal
        resistance_levels = sr_levels.get("resistance", [])
        if resistance_levels:
            target_price = min([r["price"] for r in resistance_levels if r["price"] > price], default=price * 1.1)
        else:
            target_price = price * 1.1
    else:  # Sell signal
        support_levels = sr_levels.get("support", [])
        if support_levels:
            target_price = max([s["price"] for s in support_levels if s["price"] < price], default=price * 0.9)
        else:
            target_price = price * 0.9
    
    risk_reward = calculate_risk_reward(entry_price, stop_loss, target_price)
    
    return {
        "signal": signal_type,
        "strength": min(max(score, -10), 10),  # Clamp to -10 to 10
        "score": score,
        "details": details,
        "entry_price": round(entry_price, 2),
        "stop_loss": round(stop_loss, 2),
        "target_price": round(target_price, 2),
        "risk_reward": risk_reward,
        "current_price": round(price, 2)
    }

# --- Win Rate Analytics ---
def analyze_strategy_performance(journal: list, start: date, end: date, mode: str) -> dict:
    """Analyze trading strategy performance from journal."""
    if not journal:
        return {}
    
    # Filter trades in date range
    trades = [t for t in journal if start <= pd.to_datetime(t.get("date", "2000-01-01")).date() <= end]
    
    if not trades:
        return {}
    
    # Calculate metrics
    total_trades = len(trades)
    buy_trades = [t for t in trades if t.get("type") == "Buy"]
    sell_trades = [t for t in trades if t.get("type") == "Sell"]
    
    # Calculate P&L for completed trades (pairs of buy/sell)
    completed_trades = []
    for buy in buy_trades:
        # Find corresponding sell
        buy_date = pd.to_datetime(buy.get("date", "2000-01-01"))
        matching_sells = [s for s in sell_trades 
                         if s.get("ticker") == buy.get("ticker") 
                         and pd.to_datetime(s.get("date", "2000-01-01")) > buy_date]
        
        if matching_sells:
            sell = min(matching_sells, key=lambda x: pd.to_datetime(x.get("date", "2000-01-01")))
            pnl = (sell.get("price", 0) - buy.get("price", 0)) * buy.get("shares", 0)
            pnl_pct = ((sell.get("price", 0) - buy.get("price", 0)) / buy.get("price", 0)) * 100 if buy.get("price", 0) > 0 else 0
            completed_trades.append({
                "ticker": buy.get("ticker"),
                "pnl": pnl,
                "pnl_pct": pnl_pct,
                "entry": buy.get("price", 0),
                "exit": sell.get("price", 0),
                "hold_days": (pd.to_datetime(sell.get("date", "2000-01-01")) - buy_date).days
            })
    
    if not completed_trades:
        return {
            "total_trades": total_trades,
            "completed_trades": 0,
            "win_rate": 0,
            "avg_return": 0
        }
    
    winning_trades = [t for t in completed_trades if t["pnl"] > 0]
    losing_trades = [t for t in completed_trades if t["pnl"] <= 0]
    
    win_rate = (len(winning_trades) / len(completed_trades)) * 100 if completed_trades else 0
    avg_return = sum(t["pnl_pct"] for t in completed_trades) / len(completed_trades) if completed_trades else 0
    avg_win = sum(t["pnl_pct"] for t in winning_trades) / len(winning_trades) if winning_trades else 0
    avg_loss = sum(t["pnl_pct"] for t in losing_trades) / len(losing_trades) if losing_trades else 0
    
    # Best and worst trades
    best_trade = max(completed_trades, key=lambda x: x["pnl_pct"]) if completed_trades else None
    worst_trade = min(completed_trades, key=lambda x: x["pnl_pct"]) if completed_trades else None
    
    # Performance by ticker
    ticker_performance = {}
    for trade in completed_trades:
        ticker = trade["ticker"]
        if ticker not in ticker_performance:
            ticker_performance[ticker] = {"trades": 0, "wins": 0, "total_pnl": 0}
        ticker_performance[ticker]["trades"] += 1
        if trade["pnl"] > 0:
            ticker_performance[ticker]["wins"] += 1
        ticker_performance[ticker]["total_pnl"] += trade["pnl_pct"]
    
    return {
        "total_trades": total_trades,
        "completed_trades": len(completed_trades),
        "win_rate": round(win_rate, 2),
        "avg_return": round(avg_return, 2),
        "avg_win": round(avg_win, 2),
        "avg_loss": round(avg_loss, 2),
        "best_trade": best_trade,
        "worst_trade": worst_trade,
        "ticker_performance": ticker_performance,
        "total_pnl": sum(t["pnl"] for t in completed_trades)
    }

# --- Lazy imports for data providers ---
@lru_cache(maxsize=1)
def _yf():
    import yfinance as yf
    return yf

@lru_cache(maxsize=1)
def _cg():
    from pycoingecko import CoinGeckoAPI
    return CoinGeckoAPI()

@lru_cache(maxsize=1)
def _gf():
    """Initialize Google Finance scraper."""
    import requests
    return requests

# --- Utilities ---
def _ensure_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, utc=False)
    # Ensure timezone-naive index to avoid UTC offset issues
    if isinstance(df.index, pd.DatetimeIndex) and df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    df = df.sort_index()
    return df

def _to_ohlc(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize to ['open','high','low','close','volume'].
    Robust to yfinance MultiIndex columns like ('Open','SPY') or ('SPY','Open').
    """
    import numpy as _np
    import pandas as _pd

    if df is None or len(df) == 0:
        return _pd.DataFrame(columns=["open","high","low","close","volume"])

    # Flatten MultiIndex columns by picking the OHLC token where present
    if isinstance(df.columns, _pd.MultiIndex):
        flat_cols = []
        for col in df.columns:
            parts = [str(p) for p in col if p is not None]
            tokens = [p.lower() for p in parts]
            field = None
            for p in tokens:
                if p in {"open","high","low","close","adj close","adj_close","adjclose","volume"}:
                    field = p
                    break
            if field is None:
                field = tokens[-1] if tokens else "close"
            if field == "adj close":
                field = "adj_close"
            flat_cols.append(field)
        df.columns = flat_cols
    else:
        df.columns = [str(c) for c in df.columns]

    cols_lower = [c.lower() for c in df.columns]
    out = _pd.DataFrame(index=df.index)

    def pick(*cands):
        for name in cands:
            name_l = name.lower()
            if name_l in cols_lower:
                return df.iloc[:, cols_lower.index(name_l)]
        return _pd.Series(_np.nan, index=df.index, dtype="float64")

    out["open"]   = pick("open")
    out["high"]   = pick("high")
    out["low"]    = pick("low")
    close_series  = pick("close")
    if close_series.isna().all():
        close_series = pick("adj_close","adj close","adjclose")
    out["close"]  = close_series
    out["volume"] = pick("volume")

    return out

def _sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window, min_periods=1).mean()

def _ema(series: pd.Series, window: int) -> pd.Series:
    return series.ewm(span=window, adjust=False).mean()

def _pct(a: float, b: float) -> float:
    if b == 0 or (b is None) or (a is None) or np.isnan(a) or np.isnan(b):
        return np.nan
    return (a / b - 1.0) * 100.0

# --- Technical Indicators ---
def _rsi(series: pd.Series, window: int = 14) -> pd.Series:
    """Calculate Relative Strength Index."""
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def _macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Calculate MACD (Moving Average Convergence Divergence).
    Returns: (MACD line, Signal line, Histogram)"""
    ema_fast = _ema(series, fast)
    ema_slow = _ema(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = _ema(macd_line, signal)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def _sideways_mask(price: pd.Series, window: int = 10, threshold: float = 0.08) -> pd.Series:
    """
    Detect sideways / range-bound conditions.
    We look at the rolling high-low range relative to the rolling mean price.
    If the range is below `threshold` (e.g. ~8%), we consider that window sideways.
    Returns a boolean Series aligned with `price`.
    """
    rolling_max = price.rolling(window=window, min_periods=window).max()
    rolling_min = price.rolling(window=window, min_periods=window).min()
    rolling_mean = price.rolling(window=window, min_periods=window).mean()
    with np.errstate(divide="ignore", invalid="ignore"):
        price_range_pct = (rolling_max - rolling_min) / rolling_mean
    return price_range_pct < threshold


def _macd_extended(series: pd.Series,
                   fast: int = 12,
                   slow: int = 26,
                   signal: int = 9,
                   sideways_window: int = 10,
                   sideways_threshold: float = 0.08) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    Extended MACD that flattens during sideways markets.
    - Computes standard MACD.
    - Detects sideways/range-bound conditions using `_sideways_mask`.
    - During sideways periods, MACD, signal, and histogram are forced toward 0,
      so the indicator appears flat when the market is chopping sideways.
    """
    macd_line, signal_line, histogram = _macd(series, fast=fast, slow=slow, signal=signal)
    mask = _sideways_mask(series, window=sideways_window, threshold=sideways_threshold)

    # Make copies so we don't mutate the originals
    macd_ext = macd_line.copy()
    signal_ext = signal_line.copy()
    hist_ext = histogram.copy()

    macd_ext[mask] = 0.0
    signal_ext[mask] = 0.0
    hist_ext[mask] = 0.0

    return macd_ext, signal_ext, hist_ext

def _bollinger_bands(series: pd.Series, window: int = 20, num_std: float = 2.0) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Calculate Bollinger Bands.
    Returns: (Upper band, Middle band (SMA), Lower band)"""
    middle = _sma(series, window)
    std = series.rolling(window=window).std()
    upper = middle + (std * num_std)
    lower = middle - (std * num_std)
    return upper, middle, lower

def _volume_sma(volume: pd.Series, window: int = 20) -> pd.Series:
    """Calculate volume moving average."""
    return _sma(volume, window)

def _vwap(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """Calculate Volume Weighted Average Price."""
    typical_price = (df["high"] + df["low"] + df["close"]) / 3
    vwap = (typical_price * df["volume"]).rolling(window=window).sum() / df["volume"].rolling(window=window).sum()
    return vwap

def _atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    """Calculate Average True Range."""
    high_low = df["high"] - df["low"]
    high_close = np.abs(df["high"] - df["close"].shift())
    low_close = np.abs(df["low"] - df["close"].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    return true_range.rolling(window=window).mean()

def _supertrend(df: pd.DataFrame, period: int = 10, multiplier: float = 3.0) -> tuple[pd.Series, pd.Series]:
    """Calculate Supertrend indicator.
    Returns: (supertrend_line, trend_direction) where trend_direction is 1 for uptrend, -1 for downtrend."""
    atr = _atr(df, window=period)
    hl_avg = (df["high"] + df["low"]) / 2
    
    # Initialize bands
    upper_band = hl_avg + (multiplier * atr)
    lower_band = hl_avg - (multiplier * atr)
    
    # Convert to numpy arrays for iteration
    upper_values = upper_band.values
    lower_values = lower_band.values
    close_values = df["close"].values
    
    supertrend_values = np.full(len(df), np.nan)
    trend_values = np.full(len(df), np.nan)
    
    # Calculate Supertrend
    for i in range(len(df)):
        if i == 0:
            supertrend_values[i] = upper_values[i]
            trend_values[i] = 1
            continue
        
        # Adjust upper band
        if close_values[i] <= upper_values[i-1]:
            upper_values[i] = min(upper_values[i], upper_values[i-1])
        
        # Adjust lower band
        if close_values[i] >= lower_values[i-1]:
            lower_values[i] = max(lower_values[i], lower_values[i-1])
        
        # Determine trend
        if close_values[i] <= supertrend_values[i-1]:
            supertrend_values[i] = upper_values[i]
            trend_values[i] = -1
        else:
            supertrend_values[i] = lower_values[i]
            trend_values[i] = 1
    
    supertrend = pd.Series(supertrend_values, index=df.index)
    trend = pd.Series(trend_values, index=df.index)
    
    return supertrend, trend

# --- Cross Detection and History ---
def detect_crosses(df: pd.DataFrame, s20: str = "SMA20", s200: str = "SMA200") -> list:
    """Detect all Golden and Death crosses in the dataframe. Returns list of cross events."""
    if df is None or df.empty or s20 not in df.columns or s200 not in df.columns:
        return []
    
    crosses = []
    s_prev = df[s20].shift(1)
    l_prev = df[s200].shift(1)
    cross_up = (s_prev <= l_prev) & (df[s20] > df[s200])
    cross_dn = (s_prev >= l_prev) & (df[s20] < df[s200])
    
    for idx in df.index[cross_up]:
        crosses.append({
            "date": idx.strftime("%Y-%m-%d") if isinstance(idx, pd.Timestamp) else str(idx),
            "type": "Golden",
            "price": float(df.loc[idx, "close"]),
            "sma20": float(df.loc[idx, s20]),
            "sma200": float(df.loc[idx, s200])
        })
    
    for idx in df.index[cross_dn]:
        crosses.append({
            "date": idx.strftime("%Y-%m-%d") if isinstance(idx, pd.Timestamp) else str(idx),
            "type": "Death",
            "price": float(df.loc[idx, "close"]),
            "sma20": float(df.loc[idx, s20]),
            "sma200": float(df.loc[idx, s200])
        })
    
    return sorted(crosses, key=lambda x: x["date"])

def analyze_cross_performance(df: pd.DataFrame, crosses: list, days_after: int = 30) -> list:
    """Analyze performance after each cross. Returns crosses with performance metrics."""
    if df.empty or not crosses:
        return crosses
    
    result = []
    for cross in crosses:
        try:
            cross_date = pd.to_datetime(cross["date"])
            cross_idx = df.index.get_indexer([cross_date], method="nearest")[0]
            if cross_idx < 0 or cross_idx >= len(df):
                cross["performance"] = None
                result.append(cross)
                continue
            
            # Get price at cross
            price_at_cross = df.iloc[cross_idx]["close"]
            
            # Get price N days after (or last available)
            future_idx = min(cross_idx + days_after, len(df) - 1)
            price_after = df.iloc[future_idx]["close"]
            
            # Calculate performance
            pct_change = ((price_after - price_at_cross) / price_at_cross) * 100
            days_actual = future_idx - cross_idx
            
            cross["performance"] = {
                "price_after": float(price_after),
                "pct_change": float(pct_change),
                "days_actual": int(days_actual),
                "days_target": days_after
            }
        except Exception:
            cross["performance"] = None
        
        result.append(cross)
    
    return result

# --- Alert Checking ---
def check_alerts(ticker: str, df: pd.DataFrame, alerts: list, send_email: bool = False) -> list:
    """Check if any alerts should be triggered. Returns list of triggered alerts."""
    if df.empty:
        return []
    
    triggered = []
    last = df.iloc[-1]
    price = float(last["close"])
    
    # Calculate indicators if needed
    rsi = None
    if any(a.get("type") in ["rsi_overbought", "rsi_oversold"] for a in alerts):
        rsi_series = _rsi(df["close"], window=14)
        rsi = float(rsi_series.iloc[-1]) if not pd.isna(rsi_series.iloc[-1]) else None
    
    # Load alert history to prevent duplicate notifications
    alert_history = load_alert_history()
    recent_alerts = {h.get("alert_id") for h in alert_history[-100:] if h.get("timestamp", "") > (datetime.now() - timedelta(hours=24)).isoformat()}
    
    for alert in alerts:
        if alert.get("ticker") != ticker or not alert.get("enabled", True):
            continue
        
        alert_type = alert.get("type")
        triggered_alert = None
        alert_id = f"{ticker}_{alert_type}_{alert.get('value', '')}"
        
        # Skip if already notified recently
        if alert_id in recent_alerts and not send_email:
            continue
        
        if alert_type == "golden_cross":
            d = df.copy()
            d["SMA20"] = _sma(d["close"], SMA_SHORT)
            d["SMA200"] = _sma(d["close"], SMA_LONG)
            s_prev = d["SMA20"].shift(1)
            l_prev = d["SMA200"].shift(1)
            if len(d) > 1 and s_prev.iloc[-1] <= l_prev.iloc[-1] and d["SMA20"].iloc[-1] > d["SMA200"].iloc[-1]:
                triggered_alert = {"alert": alert, "message": f"Golden Cross detected for {ticker} at ${price:.2f}"}
        
        elif alert_type == "death_cross":
            d = df.copy()
            d["SMA20"] = _sma(d["close"], SMA_SHORT)
            d["SMA200"] = _sma(d["close"], SMA_LONG)
            s_prev = d["SMA20"].shift(1)
            l_prev = d["SMA200"].shift(1)
            if len(d) > 1 and s_prev.iloc[-1] >= l_prev.iloc[-1] and d["SMA20"].iloc[-1] < d["SMA200"].iloc[-1]:
                triggered_alert = {"alert": alert, "message": f"Death Cross detected for {ticker} at ${price:.2f}"}
        
        elif alert_type == "price_above" and price >= alert.get("value", 0):
            triggered_alert = {"alert": alert, "message": f"{ticker} price ${price:.2f} is above ${alert.get('value', 0):.2f}"}
        
        elif alert_type == "price_below" and price <= alert.get("value", 0):
            triggered_alert = {"alert": alert, "message": f"{ticker} price ${price:.2f} is below ${alert.get('value', 0):.2f}"}
        
        elif alert_type == "rsi_oversold" and rsi is not None and rsi <= alert.get("value", 30):
            triggered_alert = {"alert": alert, "message": f"{ticker} RSI {rsi:.1f} is oversold (≤{alert.get('value', 30)})"}
        
        elif alert_type == "rsi_overbought" and rsi is not None and rsi >= alert.get("value", 70):
            triggered_alert = {"alert": alert, "message": f"{ticker} RSI {rsi:.1f} is overbought (≥{alert.get('value', 70)})"}
        
        if triggered_alert:
            triggered.append(triggered_alert)
            
            # Send email if configured
            if send_email and alert.get("email_enabled", False):
                email_config = load_email_config()
                to_email = alert.get("email", email_config.get("default_email", ""))
                if to_email and email_config.get("from_email"):
                    subject = f"TraderQ Alert: {ticker}"
                    body = triggered_alert["message"]
                    if send_email_alert(to_email, subject, body, email_config):
                        # Log successful email
                        alert_history.append({
                            "alert_id": alert_id,
                            "ticker": ticker,
                            "type": alert_type,
                            "message": triggered_alert["message"],
                            "timestamp": datetime.now().isoformat(),
                            "email_sent": True
                        })
                        save_alert_history(alert_history)
            
            # Log alert to history
            alert_history.append({
                "alert_id": alert_id,
                "ticker": ticker,
                "type": alert_type,
                "message": triggered_alert["message"],
                "timestamp": datetime.now().isoformat(),
                "email_sent": send_email and alert.get("email_enabled", False)
            })
            save_alert_history(alert_history)
    
    return triggered

# --- Portfolio Calculations ---
def calculate_portfolio_metrics(portfolio: dict, start: date, end: date, mode: str) -> dict:
    """Calculate portfolio-level metrics."""
    if not portfolio.get("tickers") or len(portfolio["tickers"]) == 0:
        return {"error": "No tickers in portfolio"}
    
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
        df = load_data(ticker, start, end, mode, data_source="Yahoo Finance")
        
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

# --- Backtesting ---
def backtest_strategy(df: pd.DataFrame, strategy: str = "golden_death", initial_capital: float = 10000.0) -> dict:
    """Backtest a trading strategy. Returns performance metrics."""
    if df.empty or len(df) < SMA_LONG:
        return {"error": "Insufficient data for backtesting"}
    
    df = df.copy()
    df["SMA20"] = _sma(df["close"], SMA_SHORT)
    df["SMA200"] = _sma(df["close"], SMA_LONG)
    
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

def _badge_color(trend: str) -> str:
    if trend == "Bullish":
        return "background-color:#12391a;color:#1fd16c;font-weight:600"
    if trend == "Bearish":
        return "background-color:#3a1919;color:#ff4d4f;font-weight:600"
    return ""

# --- Data loaders ---
@st.cache_data(show_spinner=False, ttl=60)  # Cache for 60 seconds (will be cleared on manual refresh)
def load_google_finance_stock(ticker: str, start: date, end: date, interval: str = "1d", 
                              force_refresh: bool = False) -> pd.DataFrame:
    """Load stock data from Google Finance - uses real-time data fetching."""
    try:
        import requests
        from bs4 import BeautifulSoup
        import re
        
        # Google Finance URL format
        quote_url = f"https://www.google.com/finance/quote/{ticker}"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
        }
        
        # Try to fetch real-time data from Google Finance
        response = requests.get(quote_url, headers=headers, timeout=15)
        
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Try to extract current price (real-time data)
            price_elem = soup.find('div', class_=re.compile('YMlKec|fxKbKc')) or \
                        soup.find('div', jsname='vWLAgc') or \
                        soup.find('div', class_='AHmHk') or \
                        soup.find('div', {'data-last-price': True})
            
            current_price = None
            if price_elem:
                try:
                    # Try to get price from text
                    price_text = price_elem.text.strip()
                    # Remove currency symbols and commas
                    price_text = re.sub(r'[^\d.]', '', price_text)
                    if price_text:
                        current_price = float(price_text)
                except (ValueError, AttributeError):
                    # Try data attribute
                    if price_elem.get('data-last-price'):
                        current_price = float(price_elem.get('data-last-price'))
            
            # Try to get historical data from Google Finance chart data
            # Google Finance uses a JSON endpoint for chart data
            chart_data_pattern = re.search(r'var\s+chartData\s*=\s*(\[.*?\]);', response.text, re.DOTALL)
            if not chart_data_pattern:
                # Try alternative pattern
                chart_data_pattern = re.search(r'chartData\s*:\s*(\[.*?\]),', response.text, re.DOTALL)
            
            if chart_data_pattern:
                try:
                    import json
                    chart_data = json.loads(chart_data_pattern.group(1))
                    if chart_data and len(chart_data) > 0:
                        # Process chart data into DataFrame
                        dates = []
                        prices = []
                        for item in chart_data:
                            if isinstance(item, list) and len(item) >= 2:
                                # Format: [timestamp, open, high, low, close, volume]
                                ts = pd.to_datetime(item[0], unit='ms')
                                # Ensure timezone-naive timestamp
                                if ts.tz is not None:
                                    ts = ts.tz_localize(None)
                                dates.append(ts)
                                if len(item) >= 5:
                                    prices.append({
                                        'open': item[1],
                                        'high': item[2],
                                        'low': item[3],
                                        'close': item[4],
                                        'volume': item[5] if len(item) > 5 else 0
                                    })
                        
                        if dates and prices:
                            df = pd.DataFrame(prices, index=dates)
                            df = df.sort_index()
                            # Ensure timezone-naive index
                            if df.index.tz is not None:
                                df.index = df.index.tz_localize(None)
                            df = df.loc[(df.index.date >= start) & (df.index.date <= end)]
                            if len(df) > 0:
                                df = _ensure_datetime_index(df)
                                df = _to_ohlc(df)
                                # Add current real-time price if available
                                if current_price and len(df) > 0:
                                    last_date = df.index[-1]
                                    today = date.today()
                                    if last_date.date() < today:
                                        # Add today's price with timezone-naive timestamp
                                        new_row = pd.DataFrame({
                                            'open': [current_price],
                                            'high': [current_price],
                                            'low': [current_price],
                                            'close': [current_price],
                                            'volume': [0]
                                        }, index=[pd.Timestamp(today)])
                                        df = pd.concat([df, new_row])
                                return df
                except (json.JSONDecodeError, ValueError, KeyError) as e:
                    import sys
                    print(f"Failed to parse Google Finance chart data: {e}", file=sys.stderr)
        
        # Fallback: Use pandas_datareader if available (Google Finance support)
        try:
            import pandas_datareader.data as web
            df = web.DataReader(ticker, 'google', start, end)
            if df is not None and len(df) > 0:
                df = _ensure_datetime_index(df)
                df = _to_ohlc(df)
                return df
        except (ImportError, Exception):
            pass
        
        # Final fallback: Use yfinance (which gets data from Yahoo Finance)
        # This ensures we always have data, even if Google Finance scraping fails
        yf = _yf()
        try:
            # For intraday intervals, limit to last 60 days
            if interval in ["1m", "2m", "5m", "15m", "30m", "60m", "90m"]:
                max_start = date.today() - timedelta(days=60)
                if start < max_start:
                    start = max_start
            
            stock = yf.Ticker(ticker)
            df = stock.history(start=start, end=(end + timedelta(days=1)), interval=interval, auto_adjust=True)
            if df is not None and len(df) > 0:
                df = _ensure_datetime_index(df)
                df = _to_ohlc(df)
                # For 5-minute interval, limit to last 500 candles if we have more
                if interval == "5m" and len(df) > 500:
                    df = df.tail(500)
                # Ensure timezone-naive index before adding real-time price
                if df.index.tz is not None:
                    df.index = df.index.tz_localize(None)
                # Try to update with real-time price if we got it
                if current_price and len(df) > 0:
                    last_date = df.index[-1]
                    today = date.today()
                    if last_date.date() < today:
                        new_row = pd.DataFrame({
                            'open': [current_price],
                            'high': [current_price],
                            'low': [current_price],
                            'close': [current_price],
                            'volume': [0]
                        }, index=[pd.Timestamp(today)])
                        df = pd.concat([df, new_row])
                return df
        except Exception:
            pass
        
        return pd.DataFrame()
    except Exception as e:
        import sys
        print(f"Google Finance load failed for {ticker}: {e}", file=sys.stderr)
        # Final fallback to yfinance
        try:
            yf = _yf()
            stock = yf.Ticker(ticker)
            df = stock.history(start=start, end=(end + timedelta(days=1)), interval=interval, auto_adjust=True)
            if df is not None and len(df) > 0:
                df = _ensure_datetime_index(df)
                df = _to_ohlc(df)
                return df
        except Exception:
            pass
        return pd.DataFrame()

@st.cache_data(show_spinner=False)
def load_stock(ticker: str, start: date, end: date, interval: str = "1d") -> pd.DataFrame:
    yf = _yf()
    
    # For intraday intervals (5m, 15m, etc.), we need to adjust the period
    # yfinance intraday data is typically available for last 60 days max
    if interval in ["1m", "2m", "5m", "15m", "30m", "60m", "90m"]:
        # Limit to last 60 days for intraday data
        max_start = date.today() - timedelta(days=60)
        if start < max_start:
            start = max_start
    
    # New yfinance (0.2.66+) uses curl_cffi and handles sessions automatically
    # Try using Ticker API first (more reliable)
    try:
        stock = yf.Ticker(ticker)  # Don't pass session - let yfinance handle it
        df = stock.history(start=start, end=(end + timedelta(days=1)), interval=interval, auto_adjust=True)
        if df is not None and len(df) > 0:
            df = _ensure_datetime_index(df)
            df = _to_ohlc(df)
            # For 5-minute interval, limit to last 500 candles if we have more
            if interval == "5m" and len(df) > 500:
                df = df.tail(500)
            return df
    except Exception as e:
        import sys
        print(f"Ticker API failed for {ticker}: {e}", file=sys.stderr)
    
    # Fallback to download API
    max_retries = 2
    for attempt in range(max_retries):
        try:
            df = yf.download(
                tickers=ticker,
                start=start.isoformat(),
                end=(end + timedelta(days=1)).isoformat(),
                interval=interval,
                auto_adjust=True,
                progress=False,
                group_by="column",
                threads=False,
            )
            if df is not None and len(df) > 0:
                df = _ensure_datetime_index(df)
                df = _to_ohlc(df)
                # For 5-minute interval, limit to last 500 candles if we have more
                if interval == "5m" and len(df) > 500:
                    df = df.tail(500)
                return df
            if attempt < max_retries - 1:
                time.sleep(1)
        except Exception as e:
            if attempt < max_retries - 1:
                import sys
                print(f"Download attempt {attempt + 1} failed for {ticker}: {e}", file=sys.stderr)
                time.sleep(2)
                continue
    
    return pd.DataFrame()

def _cg_id_for(symbol_usd: str) -> str | None:
    # crude mapping for popular coins
    symbol = symbol_usd.upper().replace("-USD", "")
    mapping = {
        "BTC": "bitcoin",
        "ETH": "ethereum",
        "SOL": "solana",
        "ADA": "cardano",
        "XRP": "ripple",
        "DOGE": "dogecoin",
    }
    return mapping.get(symbol)

@st.cache_data(show_spinner=False)
def load_crypto(ticker: str, start: date, end: date) -> pd.DataFrame:
    """Try Yahoo Finance first for -USD pairs; fallback to CoinGecko (capped to 365d)."""
    # Try yfinance for e.g. 'BTC-USD'
    try:
        df_y = load_stock(ticker, start, end)
        if len(df_y) > 0 and not df_y["close"].isna().all():
            return df_y
    except Exception:
        pass

    # CoinGecko fallback (range limited to 365 days on public plan)
    try:
        cg = _cg()
        cg_id = _cg_id_for(ticker)
        if not cg_id:
            return pd.DataFrame()

        # Respect 365d cap
        max_days = 365
        s_cap = max(end - timedelta(days=max_days), start)
        from_ts = int(datetime.combine(s_cap, datetime.min.time()).timestamp())
        to_ts = int(datetime.combine(end, datetime.min.time()).timestamp())

        payload = cg.get_coin_market_chart_range_by_id(
            id=cg_id,
            vs_currency="usd",
            from_timestamp=from_ts,
            to_timestamp=to_ts,
        )
        if not payload or "prices" not in payload:
            return pd.DataFrame()

        # Build OHLC from prices (minute/daily granularity → we’ll downsample to 1D)
        prices = payload.get("prices", [])
        df = pd.DataFrame(prices, columns=["ts", "price"])
        df["ts"] = pd.to_datetime(df["ts"], unit="ms")
        # 1D resample for OHLC
        ohlc = df.set_index("ts")["price"].resample("1D").agg(["first", "max", "min", "last"])
        ohlc.columns = ["open", "high", "low", "close"]
        ohlc["volume"] = np.nan
        ohlc = ohlc.loc[(ohlc.index.date >= start) & (ohlc.index.date <= end)]
        return ohlc
    except Exception:
        return pd.DataFrame()

@st.cache_data(show_spinner=False)
def load_data(ticker: str, start: date, end: date, mode: str, interval: str = "1d", data_source: str = "Yahoo Finance") -> pd.DataFrame:
    """Load data from specified data source."""
    if mode == "Stocks":
        if data_source == "Google Finance":
            return load_google_finance_stock(ticker, start, end, interval=interval)
        else:
            return load_stock(ticker, start, end, interval=interval)
    # Crypto doesn't support intervals in the same way, but we can resample if needed
    # Google Finance for crypto uses same approach as stocks
    if data_source == "Google Finance":
        df = load_google_finance_stock(ticker, start, end, interval=interval)
    else:
        df = load_crypto(ticker, start, end)
    if not df.empty and interval != "1d":
        # Resample crypto data to weekly/monthly
        if interval == "1wk":
            df = df.resample("W").agg({
                "open": "first", "high": "max", "low": "min", 
                "close": "last", "volume": "sum"
            }).dropna()
        elif interval == "1mo":
            df = df.resample("M").agg({
                "open": "first", "high": "max", "low": "min", 
                "close": "last", "volume": "sum"
            }).dropna()
    return df

def add_cross_markers(fig: go.Figure, df: pd.DataFrame,
                      price_col: str = "close",
                      s20: str = "SMA20",
                      s200: str = "SMA200",
                      row: int = 1, col: int = 1) -> None:
    """
    Golden cross: 20 crosses above 200 (green triangle-up)
    Death  cross: 20 crosses below 200 (red triangle-down)
    Works if df has df[price_col], df[s20], df[s200].
    """
    if df is None or df.empty:
        return
    needed = (price_col, s20, s200)
    if any(c not in df.columns for c in needed):
        return
    s_prev = df[s20].shift(1)
    l_prev = df[s200].shift(1)
    cross_up = (s_prev <= l_prev) & (df[s20] > df[s200])
    cross_dn = (s_prev >= l_prev) & (df[s20] < df[s200])
    xu, yu = df.index[cross_up], df.loc[cross_up, price_col]
    xd, yd = df.index[cross_dn], df.loc[cross_dn, price_col]
    if len(xu):
        fig.add_trace(go.Scatter(
            x=xu, y=yu, mode="markers", name="Golden Cross",
            marker=dict(symbol="triangle-up", size=15, color="#17c964",
                        line=dict(width=1, color="#0b3820")),
            hovertemplate="Golden: %{x|%Y-%m-%d}<br>Price: %{y:.2f}<extra></extra>",
            showlegend=True
        ), row=row, col=col)
    if len(xd):
        fig.add_trace(go.Scatter(
            x=xd, y=yd, mode="markers", name="Death Cross",
            marker=dict(symbol="triangle-down", size=15, color="#f31260",
                        line=dict(width=1, color="#4a0b19")),
            hovertemplate="Death: %{x|%Y-%m-%d}<br>Price: %{y:.2f}<extra></extra>",
            showlegend=True
        ), row=row, col=col)

# --- Chart builder ---
def make_chart(df: pd.DataFrame, title: str, theme: str, pretouch_pct: float | None,
               show_volume: bool = True, show_rsi: bool = True, show_macd: bool = True,
               show_bollinger: bool = True, show_sma20: bool = True, show_sma200: bool = True,
               show_ema: bool = False, show_supertrend: bool = False,
               show_support_resistance: bool = False, show_fvg: bool = False,
               macd_mode: str = "Extended",
               macd_sideways_window: int = 10,
               macd_sideways_threshold: float = 8.0,
               trade_entry: float | None = None, trade_stop_loss: float | None = None,
               trade_target: float | None = None, trade_direction: str | None = None) -> go.Figure:
    # Ensure timezone-naive index to avoid UTC offset issues in slicing operations
    if not df.empty:
        df = df.copy()  # Work with a copy to avoid modifying the original
        if isinstance(df.index, pd.DatetimeIndex) and df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        # Also ensure all datetime columns are timezone-naive
        for col in ['open', 'high', 'low', 'close']:
            if col in df.columns and hasattr(df[col].dtype, 'tz') and df[col].dtype.tz is not None:
                df[col] = df[col].dt.tz_localize(None)
    
    if df.empty:
        fig = go.Figure()
        fig.update_layout(
            template="plotly_dark" if theme == "Dark" else "plotly_white", 
            title=title,
            hoverlabel=dict(
                bgcolor="rgba(255, 255, 255, 0.95)" if theme == "Light" else "rgba(0, 0, 0, 0.9)",
                bordercolor="rgba(0, 123, 255, 0.8)" if theme == "Light" else "rgba(0, 212, 255, 0.8)",
                font_size=20,  # 1.5x the default size (~13px)
                font_family="Arial, sans-serif",
                font_color="#212529" if theme == "Light" else "white"
            )
        )
        return fig

    df = df.copy()
    # Re-check and ensure timezone-naive index after copy
    if isinstance(df.index, pd.DatetimeIndex) and df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    
    df["SMA20"] = _sma(df["close"], SMA_SHORT)
    df["SMA200"] = _sma(df["close"], SMA_LONG)
    
    # Set background colors based on theme
    if theme == "Dark":
        plot_bgcolor = "rgba(0, 0, 0, 0)"
        paper_bgcolor = "rgba(0, 0, 0, 0)"
        template = "plotly_dark"
        hover_bg = "rgba(0, 0, 0, 0.9)"
        hover_border = "rgba(0, 212, 255, 0.8)"
        hover_text = "white"
        grid_color = "rgba(128,128,128,0.2)"
        text_color = "#e8e6e3"
    else:  # Light mode
        plot_bgcolor = "#f8f9fa"
        paper_bgcolor = "#ffffff"
        template = "plotly_white"
        hover_bg = "rgba(255, 255, 255, 0.95)"
        hover_border = "rgba(0, 123, 255, 0.8)"
        hover_text = "#212529"
        grid_color = "rgba(200,200,200,0.3)"
        text_color = "#212529"
    
    # Determine number of subplots needed
    num_subplots = 1  # Main price chart
    if show_volume:
        num_subplots += 1
    if show_rsi:
        num_subplots += 1
    if show_macd:
        num_subplots += 1
    
    # Create subplots
    from plotly.subplots import make_subplots
    fig = make_subplots(
        rows=num_subplots, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.5] + [0.5 / (num_subplots - 1)] * (num_subplots - 1) if num_subplots > 1 else [1.0],
        subplot_titles=([title] + 
                       (["Volume"] if show_volume else []) +
                       (["RSI"] if show_rsi else []) +
                       (["MACD"] if show_macd else []))
    )
    
    row = 1
    
    # Add filled area between SMAs with dynamic color based on which is on top
    # This must be added BEFORE the candlestick so it appears behind
    if show_sma20 and show_sma200:
        # Ensure indices are timezone-naive for slicing
        if isinstance(df.index, pd.DatetimeIndex) and df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        
        # Create segments where SMA20 is above vs SMA200 is above
        sma20_above = df["SMA20"] > df["SMA200"]
        
        # Find transition points where the relationship changes
        transitions = (sma20_above != sma20_above.shift(1)).fillna(False)
        transition_indices = df.index[transitions].tolist()
        
        # Add start and end to transition list
        all_points = [df.index[0]] + transition_indices + [df.index[-1]]
        
        # Ensure all indices are timezone-naive - convert each to timezone-naive timestamp
        for i, idx in enumerate(all_points):
            if isinstance(idx, pd.Timestamp):
                if idx.tz is not None:
                    all_points[i] = idx.tz_localize(None)
            elif hasattr(idx, 'tz') and idx.tz is not None:
                all_points[i] = pd.Timestamp(idx).tz_localize(None)
        
        # Create segments
        for i in range(len(all_points) - 1):
            start_idx = all_points[i]
            end_idx = all_points[i + 1]
            
            # Convert to timezone-naive if needed
            if isinstance(start_idx, pd.Timestamp) and start_idx.tz is not None:
                start_idx = start_idx.tz_localize(None)
            if isinstance(end_idx, pd.Timestamp) and end_idx.tz is not None:
                end_idx = end_idx.tz_localize(None)
            
            # Use boolean indexing to avoid timezone issues
            try:
                # Use boolean mask for more robust slicing
                start_mask = df.index >= start_idx
                end_mask = df.index <= end_idx
                segment_mask = start_mask & end_mask
                
                if segment_mask.any():
                    segment_df = df.loc[segment_mask]
                else:
                    # Fallback: use searchsorted to find positions
                    try:
                        start_pos = df.index.searchsorted(start_idx, side='left')
                        end_pos = df.index.searchsorted(end_idx, side='right')
                        if start_pos < len(df) and end_pos <= len(df) and start_pos < end_pos:
                            segment_df = df.iloc[start_pos:end_pos]
                        else:
                            continue  # Skip this segment if indices are invalid
                    except (ValueError, TypeError):
                        continue  # Skip this segment
            except (KeyError, TypeError, ValueError, IndexError) as e:
                # Fallback: use boolean mask
                try:
                    mask = (df.index >= start_idx) & (df.index <= end_idx)
                    if mask.any():
                        segment_df = df.loc[mask]
                    else:
                        continue  # Skip this segment if no matching data
                except Exception:
                    continue  # Skip this segment entirely if we can't slice it
            
            if len(segment_df) < 2:
                continue
            
            # Determine which SMA is on top in this segment
            is_sma20_top = segment_df["SMA20"].iloc[0] > segment_df["SMA200"].iloc[0]
            
            # Create upper and lower bounds for fill
            upper = segment_df[["SMA20", "SMA200"]].max(axis=1)
            lower = segment_df[["SMA20", "SMA200"]].min(axis=1)
            
            # Set color based on which is on top and theme
            if is_sma20_top:
                # SMA20 on top (green/blue) - green fill
                if theme == "Light":
                    fillcolor = "rgba(40, 167, 69, 0.15)"
                    line_color = "rgba(40, 167, 69, 0.25)"
                else:
                    fillcolor = "rgba(23, 201, 100, 0.2)"
                    line_color = "rgba(23, 201, 100, 0.3)"
            else:
                # SMA200 on top (red) - red fill
                if theme == "Light":
                    fillcolor = "rgba(220, 53, 69, 0.15)"
                    line_color = "rgba(220, 53, 69, 0.25)"
                else:
                    fillcolor = "rgba(243, 18, 96, 0.2)"
                    line_color = "rgba(243, 18, 96, 0.3)"
            
            # Add filled area
            fig.add_trace(go.Scatter(
                x=segment_df.index,
                y=upper,
                mode="lines",
                line=dict(width=0),
                showlegend=False,
                hoverinfo="skip"
            ), row=row, col=1)
            
            fig.add_trace(go.Scatter(
                x=segment_df.index,
                y=lower,
                mode="lines",
                line=dict(width=0),
                fill="tonexty",
                fillcolor=fillcolor,
                showlegend=False,
                hoverinfo="skip"
            ), row=row, col=1)
    
    # Main price chart (add after fill so it appears on top)
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df["open"], high=df["high"], low=df["low"], close=df["close"],
        name="Price"
    ), row=row, col=1)
    
    # Add SMA20 line if enabled
    if show_sma20:
        sma20_color = "#0066cc" if theme == "Light" else "#4fa3ff"
        fig.add_trace(go.Scatter(
            x=df.index, y=df["SMA20"], mode="lines", name=f"SMA {SMA_SHORT}", 
            line=dict(width=2, color=sma20_color)
        ), row=row, col=1)
    
    # Add SMA200 line if enabled
    if show_sma200:
        sma200_color = "#cc0000" if theme == "Light" else "#ff6b6b"
        fig.add_trace(go.Scatter(
            x=df.index, y=df["SMA200"], mode="lines", name=f"SMA {SMA_LONG}", 
            line=dict(width=2, color=sma200_color)
        ), row=row, col=1)
    
    # Add EMA line if enabled
    if show_ema:
        ema20 = _ema(df["close"], window=20)
        ema_color = "#00a86b" if theme == "Light" else "#00ff88"
        fig.add_trace(go.Scatter(
            x=df.index, y=ema20, mode="lines", name="EMA 20",
            line=dict(width=2, color=ema_color, dash="dot")
        ), row=row, col=1)
    
    # Add Supertrend if enabled
    if show_supertrend:
        # Ensure timezone-naive index for slicing operations
        if isinstance(df.index, pd.DatetimeIndex) and df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        
        supertrend, trend = _supertrend(df, period=10, multiplier=3.0)
        # Create continuous segments for each trend period
        uptrend_color = "#28a745" if theme == "Light" else "#00ff88"
        downtrend_color = "#dc3545" if theme == "Light" else "#ff4444"
        
        # Find change points in trend
        trend_changes = (trend != trend.shift(1)).fillna(False)
        change_indices = df.index[trend_changes].tolist()
        
        # Build segments
        if len(change_indices) > 0:
            # Create segments: each segment starts at a change point or the beginning
            segment_boundaries = [df.index[0]] + change_indices + [df.index[-1]]
            legend_shown_up = False
            legend_shown_down = False
            
            for i in range(len(segment_boundaries) - 1):
                start_idx = segment_boundaries[i]
                end_idx = segment_boundaries[i + 1]
                segment_df = df.loc[start_idx:end_idx]
                
                if len(segment_df) > 1:
                    seg_trend = trend.loc[start_idx]
                    seg_color = uptrend_color if seg_trend > 0 else downtrend_color
                    
                    # Show legend only when trend type changes
                    show_legend = False
                    if seg_trend > 0 and not legend_shown_up:
                        seg_name = "Supertrend (Up)"
                        legend_shown_up = True
                        show_legend = True
                    elif seg_trend < 0 and not legend_shown_down:
                        seg_name = "Supertrend (Down)"
                        legend_shown_down = True
                        show_legend = True
                    else:
                        seg_name = ""
                    
                    fig.add_trace(go.Scatter(
                        x=segment_df.index,
                        y=supertrend.loc[segment_df.index],
                        mode="lines",
                        name=seg_name,
                        line=dict(width=2, color=seg_color),
                        showlegend=show_legend
                    ), row=row, col=1)
        else:
            # Single trend throughout
            seg_trend = trend.iloc[0]
            seg_color = uptrend_color if seg_trend > 0 else downtrend_color
            seg_name = "Supertrend (Up)" if seg_trend > 0 else "Supertrend (Down)"
            fig.add_trace(go.Scatter(
                x=df.index,
                y=supertrend,
                mode="lines",
                name=seg_name,
                line=dict(width=2, color=seg_color)
            ), row=row, col=1)

    # Bollinger Bands
    if show_bollinger:
        bb_upper, bb_middle, bb_lower = _bollinger_bands(df["close"], window=20, num_std=2.0)
        fig.add_trace(go.Scatter(
            x=df.index, y=bb_upper, mode="lines", name="BB Upper",
            line=dict(width=1, color="#888", dash="dash"), showlegend=False
        ), row=row, col=1)
        fig.add_trace(go.Scatter(
            x=df.index, y=bb_lower, mode="lines", name="BB Lower",
            line=dict(width=1, color="#888", dash="dash"), fill="tonexty",
            fillcolor="rgba(128,128,128,0.1)", showlegend=False
        ), row=row, col=1)

    # Add Fair Value Gaps (FVG)
    if show_fvg:
        fvgs = find_fair_value_gaps(df)
        for fvg in fvgs:
            # Create a rectangular area for the FVG
            gap_high = fvg["gap_high"]
            gap_low = fvg["gap_low"]
            start_date = fvg["start_date"]
            end_date = fvg["end_date"]
            
            # Choose colors based on FVG type and theme
            if fvg["type"] == "bullish":
                # Bullish FVG: green/cyan tint
                if theme == "Light":
                    fill_color = "rgba(40, 167, 69, 0.2)"
                    line_color = "rgba(40, 167, 69, 0.6)"
                else:
                    fill_color = "rgba(0, 255, 136, 0.25)"
                    line_color = "rgba(0, 255, 136, 0.7)"
            else:
                # Bearish FVG: red/pink tint
                if theme == "Light":
                    fill_color = "rgba(220, 53, 69, 0.2)"
                    line_color = "rgba(220, 53, 69, 0.6)"
                else:
                    fill_color = "rgba(255, 68, 68, 0.25)"
                    line_color = "rgba(255, 68, 68, 0.7)"
            
            # Draw the FVG as a rectangle (filled area between gap_high and gap_low)
            # Extend slightly beyond the start/end dates for visibility
            fig.add_shape(
                type="rect",
                x0=start_date,
                y0=gap_low,
                x1=end_date,
                y1=gap_high,
                fillcolor=fill_color,
                line=dict(color=line_color, width=1, dash="dot"),
                layer="below",
                row=row, col=1
            )
            
            # Add text annotation for FVG type
            mid_date = start_date + (end_date - start_date) / 2 if isinstance(end_date, pd.Timestamp) and isinstance(start_date, pd.Timestamp) else start_date
            mid_price = (gap_high + gap_low) / 2
            fig.add_annotation(
                x=mid_date,
                y=mid_price,
                text=f"FVG ({fvg['type']})",
                showarrow=False,
                font=dict(size=9, color=line_color),
                bgcolor=fill_color,
                bordercolor=line_color,
                borderwidth=1,
                opacity=0.8,
                row=row, col=1
            )

    # Add support and resistance levels
    if show_support_resistance:
        sr_levels = find_support_resistance(df)
        for level in sr_levels.get("support", []):
            fig.add_hline(
                y=level["price"],
                line_dash="dash",
                line_color="green",
                opacity=0.5,
                annotation_text=f"Support ${level['price']:.2f}",
                row=row, col=1
            )
        for level in sr_levels.get("resistance", []):
            fig.add_hline(
                y=level["price"],
                line_dash="dash",
                line_color="red",
                opacity=0.5,
                annotation_text=f"Resistance ${level['price']:.2f}",
                row=row, col=1
            )
    
    # Add trade parameters rectangle (entry, stop loss, target)
    if trade_entry is not None and trade_stop_loss is not None and trade_target is not None and trade_direction:
        # Draw rectangle covering the trade zone from start to end of chart
        start_date = df.index[0] if not df.empty else None
        end_date = df.index[-1] if not df.empty else None
        
        if start_date and end_date:
            if trade_direction == "LONG":
                # For LONG: 
                # - Red rectangle from stop_loss (below) to entry (risk zone)
                # - Green rectangle from entry to target (reward zone)
                risk_fill_color = "rgba(243, 18, 96, 0.2)" if theme == "Dark" else "rgba(220, 53, 69, 0.15)"
                risk_line_color = "#f31260" if theme == "Dark" else "#dc3545"
                reward_fill_color = "rgba(0, 255, 136, 0.2)" if theme == "Dark" else "rgba(40, 167, 69, 0.15)"
                reward_line_color = "#00ff88" if theme == "Dark" else "#28a745"
                
                # Draw red rectangle (risk zone: stop_loss to entry)
                fig.add_shape(
                    type="rect",
                    x0=start_date,
                    y0=trade_stop_loss,
                    x1=end_date,
                    y1=trade_entry,
                    fillcolor=risk_fill_color,
                    line=dict(color=risk_line_color, width=1, dash="dot"),
                    layer="below",
                    row=row, col=1
                )
                
                # Draw green rectangle (reward zone: entry to target)
                fig.add_shape(
                    type="rect",
                    x0=start_date,
                    y0=trade_entry,
                    x1=end_date,
                    y1=trade_target,
                    fillcolor=reward_fill_color,
                    line=dict(color=reward_line_color, width=1, dash="dot"),
                    layer="below",
                    row=row, col=1
                )
                
            elif trade_direction == "SHORT":
                # For SHORT:
                # - Red rectangle from entry to stop_loss (above) (risk zone)
                # - Green rectangle from target (below) to entry (reward zone)
                risk_fill_color = "rgba(243, 18, 96, 0.2)" if theme == "Dark" else "rgba(220, 53, 69, 0.15)"
                risk_line_color = "#f31260" if theme == "Dark" else "#dc3545"
                reward_fill_color = "rgba(0, 255, 136, 0.2)" if theme == "Dark" else "rgba(40, 167, 69, 0.15)"
                reward_line_color = "#00ff88" if theme == "Dark" else "#28a745"
                
                # Draw red rectangle (risk zone: entry to stop_loss)
                fig.add_shape(
                    type="rect",
                    x0=start_date,
                    y0=trade_entry,
                    x1=end_date,
                    y1=trade_stop_loss,
                    fillcolor=risk_fill_color,
                    line=dict(color=risk_line_color, width=1, dash="dot"),
                    layer="below",
                    row=row, col=1
                )
                
                # Draw green rectangle (reward zone: target to entry)
                fig.add_shape(
                    type="rect",
                    x0=start_date,
                    y0=trade_target,
                    x1=end_date,
                    y1=trade_entry,
                    fillcolor=reward_fill_color,
                    line=dict(color=reward_line_color, width=1, dash="dot"),
                    layer="below",
                    row=row, col=1
                )
            else:
                # HOLD/NEUTRAL - single gray rectangle
                rect_low = min(trade_stop_loss, trade_entry, trade_target)
                rect_high = max(trade_stop_loss, trade_entry, trade_target)
                fill_color = "rgba(176, 176, 176, 0.15)" if theme == "Dark" else "rgba(128, 128, 128, 0.1)"
                line_color = "#b0b0b0" if theme == "Dark" else "#888888"
                
                fig.add_shape(
                    type="rect",
                    x0=start_date,
                    y0=rect_low,
                    x1=end_date,
                    y1=rect_high,
                    fillcolor=fill_color,
                    line=dict(color=line_color, width=2, dash="dash"),
                    layer="below",
                    row=row, col=1
                )
            
            # Add horizontal lines for entry, stop loss, and target
            entry_line_color = "#00ff88" if theme == "Dark" else "#28a745" if trade_direction == "LONG" else "#f31260" if trade_direction == "SHORT" else "#b0b0b0"
            fig.add_hline(
                y=trade_entry,
                line_dash="solid",
                line_color=entry_line_color,
                line_width=2,
                opacity=0.8,
                annotation_text=f"Entry: ${trade_entry:.2f}",
                annotation_position="right",
                row=row, col=1
            )
            fig.add_hline(
                y=trade_stop_loss,
                line_dash="dot",
                line_color="#ff4444" if theme == "Dark" else "#dc3545",
                line_width=1.5,
                opacity=0.7,
                annotation_text=f"Stop Loss: ${trade_stop_loss:.2f}",
                annotation_position="right",
                row=row, col=1
            )
            fig.add_hline(
                y=trade_target,
                line_dash="dot",
                line_color="#00ff88" if theme == "Dark" else "#28a745",
                line_width=1.5,
                opacity=0.7,
                annotation_text=f"Target: ${trade_target:.2f}",
                annotation_position="right",
                row=row, col=1
            )

    # Add cross markers (golden/death crosses)
    add_cross_markers(fig, df, price_col="close", s20="SMA20", s200="SMA200", row=row, col=1)

    # Pretouch band (symmetric % around SMA200)
    if pretouch_pct and pretouch_pct > 0:
        band = df["SMA200"] * (pretouch_pct / 100.0)
        upper = df["SMA200"] + band
        lower = df["SMA200"] - band
        fig.add_trace(go.Scatter(
            x=df.index, y=upper, line=dict(width=0), showlegend=False, hoverinfo="skip"
        ), row=row, col=1)
        fig.add_trace(go.Scatter(
            x=df.index, y=lower, fill="tonexty", name=f"Pretouch ±{pretouch_pct:.2f}%",
            hoverinfo="skip", opacity=0.15
        ), row=row, col=1)

    # Volume subplot
    if show_volume and not df["volume"].isna().all():
        row += 1
        # Adjust volume colors for light mode
        if theme == "Light":
            colors = ["#28a745" if df["close"].iloc[i] >= df["open"].iloc[i] else "#dc3545" 
                     for i in range(len(df))]
            vol_sma_color = "#495057"
        else:
            colors = ["#17c964" if df["close"].iloc[i] >= df["open"].iloc[i] else "#f31260" 
                     for i in range(len(df))]
            vol_sma_color = "#888"
        fig.add_trace(go.Bar(
            x=df.index, y=df["volume"], name="Volume",
            marker_color=colors, opacity=0.6
        ), row=row, col=1)
        
        # Volume SMA
        vol_sma = _volume_sma(df["volume"], window=20)
        fig.add_trace(go.Scatter(
            x=df.index, y=vol_sma, mode="lines", name="Vol SMA 20",
            line=dict(width=1, color=vol_sma_color)
        ), row=row, col=1)
        
        fig.update_yaxes(title_text="Volume", row=row, col=1)

    # RSI subplot
    if show_rsi:
        row += 1
        rsi = _rsi(df["close"], window=14)
        rsi_color = "#6f42c1" if theme == "Light" else "#9b59b6"
        fig.add_trace(go.Scatter(
            x=df.index, y=rsi, mode="lines", name="RSI",
            line=dict(width=2, color=rsi_color)
        ), row=row, col=1)
        
        # RSI levels
        overbought_color = "#dc3545" if theme == "Light" else "red"
        oversold_color = "#28a745" if theme == "Light" else "green"
        neutral_color = "#6c757d" if theme == "Light" else "gray"
        fig.add_hline(y=70, line_dash="dash", line_color=overbought_color, opacity=0.5, row=row, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color=oversold_color, opacity=0.5, row=row, col=1)
        fig.add_hline(y=50, line_dash="dot", line_color=neutral_color, opacity=0.3, row=row, col=1)
        
        fig.update_yaxes(title_text="RSI", range=[0, 100], row=row, col=1)

    # MACD subplot
    if show_macd:
        row += 1
        # Choose between normal and extended MACD
        if macd_mode == "Normal":
            macd_line, signal_line, histogram = _macd(df["close"])
            macd_title = "MACD"
        else:
            macd_line, signal_line, histogram = _macd_extended(
                df["close"],
                sideways_window=macd_sideways_window,
                sideways_threshold=macd_sideways_threshold / 100.0
            )
            macd_title = "Extended MACD"
        macd_color = "#0056b3" if theme == "Light" else "#3498db"
        signal_color = "#c82333" if theme == "Light" else "#e74c3c"
        fig.add_trace(go.Scatter(
            x=df.index, y=macd_line, mode="lines", name=macd_title,
            line=dict(width=2, color=macd_color)
        ), row=row, col=1)
        fig.add_trace(go.Scatter(
            x=df.index, y=signal_line, mode="lines", name="Signal",
            line=dict(width=2, color=signal_color)
        ), row=row, col=1)
        
        # Histogram - adjust colors for light mode
        if theme == "Light":
            colors_hist = ["#28a745" if h >= 0 else "#dc3545" for h in histogram]
        else:
            colors_hist = ["#17c964" if h >= 0 else "#f31260" for h in histogram]
        fig.add_trace(go.Bar(
            x=df.index, y=histogram, name="Histogram",
            marker_color=colors_hist, opacity=0.6
        ), row=row, col=1)
        
        zero_line_color = "#6c757d" if theme == "Light" else "gray"
        fig.add_hline(y=0, line_dash="dot", line_color=zero_line_color, opacity=0.3, row=row, col=1)
        fig.update_yaxes(title_text="MACD", row=row, col=1)

    # Add cross markers (golden/death crosses)
    add_cross_markers(fig, df, price_col="close", s20="SMA20", s200="SMA200", row=1, col=1)

    fig.update_layout(
        template=template,
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(l=10, r=10, t=60, b=10),
        height=400 + (200 * (num_subplots - 1)),
        plot_bgcolor=plot_bgcolor,
        paper_bgcolor=paper_bgcolor,
        hoverlabel=dict(
            bgcolor=hover_bg,
            bordercolor=hover_border,
            font_size=20,  # 1.5x the default size (~13px)
            font_family="Arial, sans-serif",
            font_color=hover_text
        ),
        font=dict(color=text_color, size=12)
    )
    
    # Update the main chart's background color and grid
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor=grid_color, row=row, col=1)
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor=grid_color, row=row, col=1)
    
    # Update all subplot axes for light mode
    if theme == "Light":
        for i in range(1, num_subplots + 1):
            fig.update_xaxes(gridcolor=grid_color, row=i, col=1)
            fig.update_yaxes(gridcolor=grid_color, row=i, col=1)
    
    return fig

# --- Screener helper ---
def build_screener(tickers: list[str], start: date, end: date, mode: str, pretouch_pct: float | None, interval: str = "1d") -> pd.DataFrame:
    rows = []
    for t in tickers:
        d = load_data(t, start, end, mode, interval=interval, data_source="Yahoo Finance")
        if d.empty or d["close"].isna().all():
            continue
        d = d.copy()
        d["SMA20"] = _sma(d["close"], SMA_SHORT)
        d["SMA200"] = _sma(d["close"], SMA_LONG)
        last = d.iloc[-1]

        price = float(last["close"])
        sma20 = float(last["SMA20"])
        sma200 = float(last["SMA200"])

        trend = ("Bullish" if (not math.isnan(sma20) and not math.isnan(sma200) and sma20 > sma200)
                 else ("Bearish" if (not math.isnan(sma20) and not math.isnan(sma200)) else "-"))

        row = {
            "Ticker": t,
            "Trend": trend,
            "Last": price,
            "SMA20": sma20,
            "SMA200": sma200,
            "Dist to SMA200 (%)": _pct(price, sma200),
            "Dist to SMA20 (%)": _pct(price, sma20),
        }

        # Check if SMA20 is touching the SMA200 pretouch band
        touch = ""
        if pretouch_pct and pretouch_pct > 0 and not math.isnan(sma20) and not math.isnan(sma200):
            band = sma200 * (pretouch_pct / 100.0)
            upper = sma200 + band
            lower = sma200 - band
            if lower <= sma20 <= upper:
                touch = "🟡"  # Yellow circle when SMA20 is in the band
        row["Touch"] = touch

        # Add RSI
        rsi = _rsi(d["close"], window=14)
        rsi_last_val = rsi.iloc[-1]
        rsi_last = float(rsi_last_val) if not (pd.isna(rsi_last_val) or math.isnan(rsi_last_val)) else np.nan
        row["RSI"] = rsi_last

        # Add MACD
        macd_line, signal_line, histogram = _macd(d["close"])
        macd_last_val = macd_line.iloc[-1]
        signal_last_val = signal_line.iloc[-1]
        hist_last_val = histogram.iloc[-1]
        macd_last = float(macd_last_val) if not (pd.isna(macd_last_val) or math.isnan(macd_last_val)) else np.nan
        signal_last = float(signal_last_val) if not (pd.isna(signal_last_val) or math.isnan(signal_last_val)) else np.nan
        hist_last = float(hist_last_val) if not (pd.isna(hist_last_val) or math.isnan(hist_last_val)) else np.nan
        row["MACD"] = macd_last
        row["MACD Signal"] = signal_last
        row["MACD Hist"] = hist_last

        # Add Volume metrics
        if not d["volume"].isna().all():
            vol_last = float(last["volume"])
            vol_sma = _volume_sma(d["volume"], window=20)
            vol_sma_last_val = vol_sma.iloc[-1]
            vol_sma_last = float(vol_sma_last_val) if not (pd.isna(vol_sma_last_val) or math.isnan(vol_sma_last_val)) else np.nan
            row["Volume"] = vol_last
            row["Vol SMA20"] = vol_sma_last
            row["Vol Ratio"] = vol_last / vol_sma_last if vol_sma_last > 0 and not math.isnan(vol_sma_last) else np.nan
        else:
            row["Volume"] = np.nan
            row["Vol SMA20"] = np.nan
            row["Vol Ratio"] = np.nan

        # Add Bollinger Bands position
        bb_upper, bb_middle, bb_lower = _bollinger_bands(d["close"], window=20, num_std=2.0)
        bb_upper_last_val = bb_upper.iloc[-1]
        bb_lower_last_val = bb_lower.iloc[-1]
        bb_upper_last = float(bb_upper_last_val) if not (pd.isna(bb_upper_last_val) or math.isnan(bb_upper_last_val)) else np.nan
        bb_lower_last = float(bb_lower_last_val) if not (pd.isna(bb_lower_last_val) or math.isnan(bb_lower_last_val)) else np.nan
        if not math.isnan(bb_upper_last) and not math.isnan(bb_lower_last):
            bb_width = bb_upper_last - bb_lower_last
            bb_position = ((price - bb_lower_last) / bb_width * 100) if bb_width > 0 else np.nan
            row["BB Position %"] = bb_position
        else:
            row["BB Position %"] = np.nan

        rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df.sort_values(by=["Dist to SMA200 (%)"], key=lambda s: s.abs()).reset_index(drop=True)
    return df

# --- UI ---
# Wrap everything in error handling to catch any startup crashes
# Write to stderr immediately to ensure we can see startup progress
sys.stderr.write("=" * 70 + "\n")
sys.stderr.write("APP.PY STARTING - SET_PAGE_CONFIG\n")
sys.stderr.write("=" * 70 + "\n")
sys.stderr.flush()

try:
    sys.stderr.write("Calling st.set_page_config...\n")
    sys.stderr.flush()
    st.set_page_config(
        page_title="TraderQ - Professional Trading Analytics",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    sys.stderr.write("✓ st.set_page_config succeeded\n")
    sys.stderr.flush()
except Exception as e:
    sys.stderr.write(f"✗ st.set_page_config failed: {e}\n")
    import traceback
    traceback.print_exc(file=sys.stderr)
    sys.stderr.flush()
    # If set_page_config fails, we're in trouble - show error and stop
    st.error(f"⚠️ Failed to initialize Streamlit: {e}")
    st.stop()

# Custom CSS for high-tech professional look
st.markdown("""
<style>
    /* Main styling */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    /* Headers */
    h1 {
        color: #00d4ff;
        font-weight: 700;
        letter-spacing: -0.5px;
        border-bottom: 2px solid #00d4ff;
        padding-bottom: 0.5rem;
        margin-bottom: 1.5rem;
    }
    
    h2 {
        color: #ffffff;
        font-weight: 600;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }
    
    h3 {
        color: #b0b0b0;
        font-weight: 500;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0a0e27 0%, #1a1f3a 100%);
    }
    
    [data-testid="stSidebar"] .css-1d391kg {
        padding-top: 2rem;
    }
    
    /* Metrics styling */
    [data-testid="stMetricValue"] {
        color: #00d4ff;
        font-size: 2rem;
        font-weight: 700;
    }
    
    [data-testid="stMetricDelta"] {
        color: #00ff88;
    }
    
    /* Buttons - More specific selectors to ensure styles apply */
    .stButton > button,
    button[data-testid="baseButton-secondary"],
    button[data-testid="baseButton-primary"],
    button.stDownloadButton,
    div[data-testid="stDownloadButton"] > button {
        background: linear-gradient(90deg, #00d4ff 0%, #0099cc 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 0.78125rem 2.34375rem !important;  /* Increased by another 25%: 0.625*1.25=0.78125, 1.875*1.25=2.34375 */
        font-weight: 600 !important;
        font-size: 2em !important;  /* Increased to 2em for better visibility */
        line-height: 1.5 !important;
        min-height: 3rem !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 15px rgba(0, 212, 255, 0.3) !important;
    }
    
    .stButton > button:hover,
    button[data-testid="baseButton-secondary"]:hover,
    button[data-testid="baseButton-primary"]:hover,
    button.stDownloadButton:hover,
    div[data-testid="stDownloadButton"] > button:hover {
        background: linear-gradient(90deg, #00ff88 0%, #00d4ff 100%) !important;
        box-shadow: 0 6px 20px rgba(0, 212, 255, 0.5) !important;
        transform: translateY(-2px) !important;
    }
    
    /* Ensure buttons don't look selected/active by default */
    .stButton > button:not(:active):not(:focus),
    button[data-testid="baseButton-secondary"]:not(:active):not(:focus),
    button[data-testid="baseButton-primary"]:not(:active):not(:focus) {
        background: linear-gradient(90deg, #00d4ff 0%, #0099cc 100%) !important;
    }
    
    /* Active/focus state styling */
    .stButton > button:active,
    .stButton > button:focus,
    button[data-testid="baseButton-secondary"]:active,
    button[data-testid="baseButton-secondary"]:focus,
    button[data-testid="baseButton-primary"]:active,
    button[data-testid="baseButton-primary"]:focus {
        background: linear-gradient(90deg, #0099cc 0%, #0077aa 100%) !important;
        box-shadow: 0 2px 10px rgba(0, 212, 255, 0.4) !important;
        transform: translateY(0) !important;
    }
    
    /* Selectbox and inputs */
    .stSelectbox > div > div {
        background-color: #1a1f3a;
        border: 1px solid #00d4ff;
        border-radius: 6px;
    }
    
    /* Number inputs */
    input[type="number"] {
        background-color: #1a1f3a !important;
        border: 1px solid #00d4ff !important;
        color: #ffffff !important;
        border-radius: 6px !important;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #0a0e27;
        padding: 0.5rem;
        border-radius: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: #1a1f3a;
        color: #b0b0b0;
        border-radius: 6px;
        padding: 0.75rem 1.5rem;
        font-weight: 500;
        transition: all 0.3s ease;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(90deg, #00d4ff 0%, #0099cc 100%);
        color: white;
        box-shadow: 0 4px 15px rgba(0, 212, 255, 0.3);
    }
    
    /* Expanders */
    .streamlit-expanderHeader {
        background-color: #1a1f3a;
        border: 1px solid #00d4ff;
        border-radius: 6px;
        padding: 0.75rem;
        font-weight: 500;
    }
    
    /* Info boxes */
    .stInfo {
        background: linear-gradient(135deg, rgba(0, 212, 255, 0.1) 0%, rgba(0, 153, 204, 0.1) 100%);
        border-left: 4px solid #00d4ff;
        border-radius: 6px;
        padding: 1rem;
    }
    
    /* Success boxes */
    .stSuccess {
        background: linear-gradient(135deg, rgba(0, 255, 136, 0.1) 0%, rgba(0, 212, 255, 0.1) 100%);
        border-left: 4px solid #00ff88;
        border-radius: 6px;
    }
    
    /* Warning boxes */
    .stWarning {
        background: linear-gradient(135deg, rgba(255, 193, 7, 0.1) 0%, rgba(255, 152, 0, 0.1) 100%);
        border-left: 4px solid #ffc107;
        border-radius: 6px;
    }
    
    /* Error boxes */
    .stError {
        background: linear-gradient(135deg, rgba(243, 18, 96, 0.1) 0%, rgba(200, 0, 0, 0.1) 100%);
        border-left: 4px solid #f31260;
        border-radius: 6px;
    }
    
    /* Dataframes */
    .dataframe {
        background-color: #1a1f3a;
        border-radius: 8px;
        border: 1px solid #00d4ff;
    }
    
    /* Radio buttons */
    [data-baseweb="radio"] {
        background-color: #1a1f3a;
    }
    
    /* Sliders */
    .stSlider > div > div {
        background-color: #1a1f3a;
    }
    
    /* Checkboxes */
    [data-baseweb="checkbox"] {
        background-color: #1a1f3a;
    }
    
    /* Divider */
    hr {
        border: none;
        border-top: 2px solid #00d4ff;
        opacity: 0.3;
        margin: 2rem 0;
    }
    
    /* Code blocks */
    .stCodeBlock {
        background-color: #0a0e27;
        border: 1px solid #00d4ff;
        border-radius: 6px;
    }
    
    /* Caption styling */
    .stCaption {
        color: #b0b0b0;
        font-size: 0.85rem;
    }
    
    /* Main container background */
    .main {
        background: linear-gradient(180deg, #0a0e27 0%, #1a1f3a 50%, #0a0e27 100%);
    }
    
    /* Scrollbar styling */
    ::-webkit-scrollbar {
        width: 10px;
        height: 10px;
    }
    
    ::-webkit-scrollbar-track {
        background: #0a0e27;
    }
    
    ::-webkit-scrollbar-thumb {
        background: #00d4ff;
        border-radius: 5px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #00ff88;
    }
</style>
""", unsafe_allow_html=True)

sys.stderr.write("AFTER CSS - before glow patch\n")
sys.stderr.flush()

# Apply glow patch if available
if ui_glow_patch is not None:
    try:
        sys.stderr.write("Applying glow patch...\n")
        sys.stderr.flush()
        ui_glow_patch.apply()  # apply glow after set_page_config
        sys.stderr.write("Glow patch applied successfully\n")
        sys.stderr.flush()
    except Exception as e:
        sys.stderr.write(f"Warning: Could not apply ui_glow_patch: {e}\n")
        sys.stderr.flush()
        import traceback
        traceback.print_exc(file=sys.stderr)
        sys.stderr.flush()
else:
    sys.stderr.write("ui_glow_patch is None, skipping\n")
    sys.stderr.flush()

# ============================================================================
# Firebase Authentication - Require login before app access
# ============================================================================
# Debug: Write to console (visible in Streamlit Cloud logs)
sys.stderr.write("=" * 70 + "\n")
print("STARTING FIREBASE INITIALIZATION", file=sys.stderr)
print("=" * 70, file=sys.stderr)

# Show a simple status message first to ensure the app is responding
st.info("🔄 Initializing Firebase authentication...")
print("✓ Status message displayed", file=sys.stderr)

# Lazy import Firebase modules (after st.set_page_config)
# Wrap in outer try-catch to ensure we always show something useful
# IMPORTANT: global declaration must be at function/module level, not inside try
try:
    try:
        print("Attempting to import Firebase modules...", file=sys.stderr)
        # Import here to avoid import-time errors
        from firebase_auth import require_authentication
        from firebase_db import FirestoreDB
        print("✓ Firebase modules imported", file=sys.stderr)
        
        # Update module-level global variables (declared at top of file)
        # Note: auth, user_id, db are already declared as globals at module level
        
        print("Calling require_authentication()...", file=sys.stderr)
        auth = require_authentication()  # This will show login UI if not authenticated
        print("✓ Authentication required", file=sys.stderr)
        
        user_id = auth.get_user_id()
        user_email = auth.get_user_email()
        display_name = auth.get_display_name()
        print(f"✓ User: {display_name} ({user_email})", file=sys.stderr)
        
        # Initialize Firestore DB
        print("Initializing Firestore DB...", file=sys.stderr)
        db = FirestoreDB()
        print("✓ Firestore DB initialized", file=sys.stderr)
        
        # Clear the status message since we succeeded
        st.empty()
        print("=" * 70, file=sys.stderr)
        print("FIREBASE INITIALIZATION SUCCESSFUL", file=sys.stderr)
        print("=" * 70, file=sys.stderr)
        
    except FileNotFoundError as e:
        print(f"✗ FileNotFoundError: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        # Firebase secrets not configured - show helpful error but don't crash
        st.error("🔥 **Firebase Configuration Missing**")
        st.error("The app requires Firebase to be configured for cloud deployment.")
        st.error(f"**Error:** `{str(e)}`")
        st.info("""
        **To fix this:**
        
        1. Go to your Firebase Console: https://console.firebase.google.com/
        2. Download your service account key (Project Settings → Service Accounts)
        3. Run `python convert_key_to_toml.py` locally
        4. Copy the contents of `.streamlit_secrets_toml.txt`
        5. Go to Streamlit Cloud → Your App → Settings → Secrets
        6. Paste the TOML content and save
        
        **Important:** After saving secrets, wait 2-3 minutes for the app to redeploy.
        """)
        st.stop()
        
    except Exception as e:
        # Other Firebase errors - log but don't crash on health check
        error_msg = str(e)
        error_type = type(e).__name__
        print(f"✗ {error_type}: {error_msg}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        st.error(f"🔥 **Firebase Initialization Error** ({error_type})")
        st.error(f"**Error:** {error_msg}")
        st.info("""
        **Possible causes:**
        1. Firebase secrets are misconfigured in Streamlit Cloud
        2. Service account key is invalid or expired
        3. Firebase project permissions issue
        4. Network connectivity issue
        
        **Check your Streamlit Cloud secrets at:**
        https://share.streamlit.io → Your App → Settings → Secrets
        """)
        import traceback
        with st.expander("📋 Technical Error Details"):
            st.code(traceback.format_exc())
        st.stop()

except Exception as outer_e:
    # Ultimate fallback - catch ANY error during startup
    error_type = type(outer_e).__name__
    error_msg = str(outer_e)
    print("=" * 70, file=sys.stderr)
    print(f"CRITICAL ERROR: {error_type}", file=sys.stderr)
    print(f"Message: {error_msg}", file=sys.stderr)
    import traceback
    traceback.print_exc(file=sys.stderr)
    print("=" * 70, file=sys.stderr)
    st.error(f"💥 **Critical Startup Error** ({error_type})")
    st.error(f"The app encountered an unexpected error: {error_msg}")
    with st.expander("📋 Full Error Traceback"):
        st.code(traceback.format_exc())
    st.info("Please check the Streamlit Cloud logs for more details.")
    st.stop()

# Professional header
st.markdown(f"""
<div style="text-align: center; padding: 2rem 0; background: linear-gradient(135deg, rgba(0, 212, 255, 0.1) 0%, rgba(0, 153, 204, 0.1) 100%); border-radius: 12px; margin-bottom: 2rem; border: 1px solid rgba(0, 212, 255, 0.3);">
    <h1 style="color: #00d4ff; font-size: 3rem; font-weight: 800; margin: 0; text-shadow: 0 0 20px rgba(0, 212, 255, 0.5); letter-spacing: -1px;">
        📈 TraderQ
    </h1>
    <p style="color: #b0b0b0; font-size: 1.2rem; margin: 0.5rem 0 0 0; font-weight: 300;">
        Professional Trading Analytics Platform
    </p>
    <p style="color: #00ff88; font-size: 0.9rem; margin: 0.5rem 0 0 0; font-weight: 500;">
        Version {APP_VERSION} • Advanced Technical Analysis & Risk Management
    </p>
</div>
""", unsafe_allow_html=True)

# Main tabs
tab_easy, tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10, tab11, tab12 = st.tabs([
    "🚦 Easy Mode", "📈 Tracker", "🔔 Alerts", "📊 Cross History", "💼 Portfolio", "🧪 Backtesting",
    "📰 News", "📐 Patterns", "🔗 Correlation", "📝 Journal", "⚡ Signals", "🛡️ Risk", "🔷 Kraken"
])



# Sidebar controls with professional styling
# User profile section
st.sidebar.markdown(f"""
<div style="background: linear-gradient(135deg, rgba(0, 212, 255, 0.1) 0%, rgba(0, 153, 204, 0.1) 100%); padding: 1rem; border-radius: 8px; margin-bottom: 1rem; border: 1px solid rgba(0, 212, 255, 0.3);">
    <div style="color: #00d4ff; font-size: 0.85rem; font-weight: 600; margin-bottom: 0.3rem;">👤 Logged in as</div>
    <div style="color: #ffffff; font-size: 1rem; font-weight: 500;">{display_name}</div>
    <div style="color: #b0b0b0; font-size: 0.75rem; margin-top: 0.2rem;">{user_email}</div>
</div>
""", unsafe_allow_html=True)

if st.sidebar.button("🚪 Logout", use_container_width=True):
    auth.logout()
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.markdown("### ⚙️ Configuration")

# Initialize and restore mode from session state
if "saved_mode" not in st.session_state:
    st.session_state["saved_mode"] = "Stocks"
mode = st.sidebar.radio("Market", ["Stocks", "Crypto"], 
                       index=0 if st.session_state["saved_mode"] == "Stocks" else 1,
                       horizontal=True, key="mode_radio")
st.session_state["saved_mode"] = mode

# Initialize and restore data_source from session state
if "saved_data_source" not in st.session_state:
    st.session_state["saved_data_source"] = "Yahoo Finance"
data_source_options = ["Yahoo Finance", "Google Finance"]
default_ds_index = 0 if st.session_state["saved_data_source"] == "Yahoo Finance" else 1
data_source = st.sidebar.selectbox("Data Source", data_source_options, 
                                   index=default_ds_index,
                                   help="Select data source. Google Finance provides real-time data.",
                                   key="data_source_select")
st.session_state["saved_data_source"] = data_source

# Auto-refresh option for Google Finance
auto_refresh_enabled = False
# Initialize refresh interval in session state
if 'refresh_interval_minutes' not in st.session_state:
    st.session_state['refresh_interval_minutes'] = 5

if data_source == "Google Finance":
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🔄 Auto-Refresh")
    
    # Refresh interval setting
    refresh_interval_minutes = st.sidebar.number_input(
        "Refresh Interval (minutes)",
        min_value=1,
        max_value=60,
        value=st.session_state.get('refresh_interval_minutes', 5),
        step=1,
        help="Set how often to automatically refresh data",
        key='refresh_interval_input'
    )
    st.session_state['refresh_interval_minutes'] = refresh_interval_minutes
    refresh_interval = refresh_interval_minutes * 60  # Convert to seconds
    
    auto_refresh_enabled = st.sidebar.checkbox(
        f"🔄 Enable auto-refresh ({refresh_interval_minutes} min)", 
        value=st.session_state.get('auto_refresh_enabled', False),
        help=f"Automatically refresh data every {refresh_interval_minutes} minutes when enabled",
        key='auto_refresh_checkbox'
    )
    st.session_state['auto_refresh_enabled'] = auto_refresh_enabled
    
    # Manual refresh button
    if st.sidebar.button("🔄 Refresh Now", use_container_width=True):
        # Clear cache for Google Finance data
        load_google_finance_stock.clear()
        load_data.clear()
        # Update last refresh time
        st.session_state['last_refresh_time'] = time.time()
        st.rerun()
else:
    # Default refresh interval when not using Google Finance
    refresh_interval = 300  # 5 minutes in seconds
    
    # Initialize last refresh time if not set
    if 'last_refresh_time' not in st.session_state:
        st.session_state['last_refresh_time'] = time.time()
    
    # Auto-refresh logic
    if auto_refresh_enabled:
        current_time = time.time()
        time_since_refresh = current_time - st.session_state.get('last_refresh_time', 0)
        
        if time_since_refresh >= refresh_interval:
            # Time to refresh - only clear cache and reload data, preserve all settings
            load_google_finance_stock.clear()
            load_data.clear()
            st.session_state['last_refresh_time'] = current_time
            # Use st.rerun() which preserves session state
            st.rerun()
        
        # Show countdown timer (updates via page reloads every 10 seconds)
        remaining_seconds = max(0, int(refresh_interval - time_since_refresh))
        remaining_minutes = remaining_seconds // 60
        remaining_secs = remaining_seconds % 60
        
        # Display countdown in sidebar with progress bar
        progress_value = max(0.0, min(1.0, remaining_seconds / refresh_interval)) if refresh_interval > 0 else 0.0
        countdown_text = f"⏱️ Next refresh in: **{remaining_minutes}:{remaining_secs:02d}**"
        st.sidebar.markdown(countdown_text)
        
        # Progress bar - display with time remaining
        progress_bar = st.sidebar.progress(progress_value)
        # Add text label using HTML for better visibility
        st.sidebar.markdown(
            f'<div style="text-align: center; color: #00d4ff; font-size: 0.75rem; margin-top: -0.5rem; margin-bottom: 0.5rem;">'
            f'{remaining_minutes}:{remaining_secs:02d} remaining</div>',
            unsafe_allow_html=True
        )
        st.sidebar.caption("(Auto-updates every 10 seconds)")
        
        # Use meta refresh for automatic page reload every 10 seconds
        # This updates the countdown display and eventually triggers data refresh
        # Meta refresh preserves session state automatically
        update_interval = 10  # Reload page every 10 seconds to update countdown
        if remaining_seconds > update_interval:
            # Meta refresh tag - works at browser level
            meta_refresh = f'<meta http-equiv="refresh" content="{update_interval}">'
            st.markdown(meta_refresh, unsafe_allow_html=True)
        else:
            # Less than 10 seconds remaining, reload when time is up
            meta_refresh = f'<meta http-equiv="refresh" content="{remaining_seconds}">'
            st.markdown(meta_refresh, unsafe_allow_html=True)

# Ticker selection with persistent custom tickers (moved to sidebar)
universe = DEFAULT_STOCKS if mode == "Stocks" else DEFAULT_CRYPTOS

# Initialize session state for custom tickers per mode (load from file on first run)
if "custom_tickers_loaded" not in st.session_state:
    st.session_state["custom_tickers_loaded"] = True
    saved_data = load_custom_tickers()
    st.session_state["custom_tickers_Stocks"] = saved_data.get("custom", {}).get("Stocks", [])
    st.session_state["custom_tickers_Crypto"] = saved_data.get("custom", {}).get("Crypto", [])
    st.session_state["selected_tickers_Stocks"] = saved_data.get("selected", {}).get("Stocks", [])
    st.session_state["selected_tickers_Crypto"] = saved_data.get("selected", {}).get("Crypto", [])

if f"custom_tickers_{mode}" not in st.session_state:
    st.session_state[f"custom_tickers_{mode}"] = []
if f"selected_tickers_{mode}" not in st.session_state:
    st.session_state[f"selected_tickers_{mode}"] = []

# Ticker selection in sidebar
st.sidebar.markdown("### 📊 Symbols")
# Combine universe with custom tickers for options
all_options = list(dict.fromkeys(universe + st.session_state[f"custom_tickers_{mode}"]))
# Use saved selected tickers if available, otherwise default to universe
default_selected = st.session_state[f"selected_tickers_{mode}"] if st.session_state[f"selected_tickers_{mode}"] else universe
selected = st.sidebar.multiselect("Choose tickers", options=all_options, default=default_selected, key=f"choose_{mode}")

# Save selected tickers whenever they change
if selected != st.session_state.get(f"_prev_selected_{mode}", []):
    st.session_state[f"_prev_selected_{mode}"] = selected
    st.session_state[f"selected_tickers_{mode}"] = selected
    save_custom_tickers({
        "custom": {
            "Stocks": st.session_state.get("custom_tickers_Stocks", []),
            "Crypto": st.session_state.get("custom_tickers_Crypto", [])
        },
        "selected": {
            "Stocks": st.session_state.get("selected_tickers_Stocks", []),
            "Crypto": st.session_state.get("selected_tickers_Crypto", [])
        }
    })

# Quick Add in sidebar
st.sidebar.markdown("---")
st.sidebar.markdown("### ➕ Quick Add")
custom = st.sidebar.text_input("Add ticker", value="", placeholder="e.g., AAPL or BTC-USD", key=f"quick_add_{mode}")
if st.sidebar.button("Add", key=f"add_btn_{mode}"):
    if custom.strip():
        custom_upper = custom.strip().upper()
        if custom_upper not in st.session_state[f"custom_tickers_{mode}"]:
            st.session_state[f"custom_tickers_{mode}"].append(custom_upper)
            # Also add to selected tickers
            if custom_upper not in st.session_state[f"selected_tickers_{mode}"]:
                st.session_state[f"selected_tickers_{mode}"].append(custom_upper)
            # Save to file
            save_custom_tickers({
                "custom": {
                    "Stocks": st.session_state.get("custom_tickers_Stocks", []),
                    "Crypto": st.session_state.get("custom_tickers_Crypto", [])
                },
                "selected": {
                    "Stocks": st.session_state.get("selected_tickers_Stocks", []),
                    "Crypto": st.session_state.get("selected_tickers_Crypto", [])
                }
            })
            st.rerun()

# Initialize and restore theme from session state
if "saved_theme" not in st.session_state:
    st.session_state["saved_theme"] = "Dark"
theme_index = 0 if st.session_state["saved_theme"] == "Dark" else 1
theme = st.sidebar.radio("Chart Theme", ["Dark", "Light"], 
                        index=theme_index, horizontal=True, key="theme_radio")
st.session_state["saved_theme"] = theme

# Initialize and restore timeframe from session state
if "saved_timeframe" not in st.session_state:
    st.session_state["saved_timeframe"] = "Daily"
timeframe_options = ["5 Minutes", "Daily", "Weekly", "Monthly"]
default_timeframe_index = timeframe_options.index(st.session_state["saved_timeframe"]) if st.session_state["saved_timeframe"] in timeframe_options else 1
timeframe = st.sidebar.selectbox("Timeframe", timeframe_options, 
                                 index=default_timeframe_index, key="timeframe_select")
st.session_state["saved_timeframe"] = timeframe
interval_map = {"5 Minutes": "5m", "Daily": "1d", "Weekly": "1wk", "Monthly": "1mo"}
interval = interval_map[timeframe]
pretouch = st.sidebar.slider("Pretouch band around SMA200 (%)", min_value=0.0, max_value=5.0, value=1.0, step=0.25)
period_days = st.sidebar.select_slider("Lookback (days)", options=[180, 365, 540, 730], value=365)

# Indicator toggles
st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Indicators")
show_sma20 = st.sidebar.checkbox("Show SMA 20", value=True)
show_sma200 = st.sidebar.checkbox("Show SMA 200", value=True)
show_ema = st.sidebar.checkbox("Show EMA", value=False)
show_volume = st.sidebar.checkbox("Show Volume", value=False)
show_rsi = st.sidebar.checkbox("Show RSI", value=True)
show_macd = st.sidebar.checkbox("Show MACD", value=True)

# MACD mode selector (only shown if MACD enabled)
macd_mode = "Extended"
macd_sideways_window = 10
macd_sideways_threshold = 8.0
if show_macd:
    macd_mode = st.sidebar.radio(
        "MACD Mode",
        options=["Normal", "Extended"],
        index=1,  # Default to Extended
        help="Normal = standard MACD. Extended = MACD that flattens during sideways/ranging markets."
    )
    
    if macd_mode == "Extended":
        macd_sideways_window = st.sidebar.slider(
            "Extended MACD Lookback (bars)",
            min_value=5, max_value=50, value=10, step=1,
            help="Number of bars used to detect sideways/range-bound conditions."
        )
        macd_sideways_threshold = st.sidebar.slider(
            "Extended MACD Range Threshold (%)",
            min_value=1.0, max_value=30.0, value=8.0, step=0.5,
            help="If high-low range over the lookback is below this %, MACD is flattened."
        )

show_bollinger = st.sidebar.checkbox("Show Bollinger Bands", value=False)
show_supertrend = st.sidebar.checkbox("Show Supertrend", value=True)
show_support_resistance = st.sidebar.checkbox("Show Support/Resistance", value=False)
show_fvg = st.sidebar.checkbox("Show Fair Value Gap", value=True)

# Date range - adjust for intraday intervals to get enough data points
end_d = date.today()
if interval == "5m":
    # For 5-minute data, we need at least 500 candles
    # 500 candles * 5 minutes = 2500 minutes = ~41.67 hours = ~1.74 days
    # Add some buffer and round up to 3 days to ensure we get enough data
    # Also account for market hours (6.5 hours/day for US market = ~8 trading days)
    start_d = end_d - timedelta(days=8)  # Get last 8 days to ensure ~500 5-min candles
elif interval in ["1m", "2m", "15m", "30m", "60m", "90m"]:
    # Other intraday intervals - calculate days needed for ~500 candles
    minutes_per_candle = {"1m": 1, "2m": 2, "15m": 15, "30m": 30, "60m": 60, "90m": 90}.get(interval, 60)
    days_needed = max(2, (500 * minutes_per_candle) / (390 * 60))  # 390 minutes per trading day
    start_d = end_d - timedelta(days=int(days_needed) + 2)  # Add buffer
else:
    # Daily, weekly, monthly intervals use the period_days slider
    start_d = end_d - timedelta(days=int(period_days))

# Watchlists section in main area
st.markdown("### 📋 Watchlists")
watchlists = load_watchlists()
watchlist_names = list(watchlists.keys())

# Load watchlist selector
if watchlist_names:
    selected_watchlist = st.selectbox("Select Watchlist", ["-- New --"] + watchlist_names, key=f"watchlist_select_{mode}")
    if selected_watchlist != "-- New --":
        if st.button("Load Watchlist", key=f"load_watchlist_{mode}"):
            st.session_state[f"selected_tickers_{mode}"] = watchlists[selected_watchlist].get("tickers", [])
            st.rerun()
        if st.button("Delete Watchlist", key=f"delete_watchlist_{mode}"):
            del watchlists[selected_watchlist]
            save_watchlists(watchlists)
            st.rerun()

# Create new watchlist
new_watchlist_name = st.text_input("New Watchlist Name", key=f"new_watchlist_{mode}", placeholder="e.g., Tech Stocks")
if st.button("Create Watchlist", key=f"create_watchlist_{mode}"):
    if new_watchlist_name.strip() and selected:
        watchlists[new_watchlist_name.strip()] = {
            "tickers": selected.copy(),
            "mode": mode,
            "created": date.today().isoformat()
        }
        save_watchlists(watchlists)
        st.success(f"Watchlist '{new_watchlist_name.strip()}' created!")
        st.rerun()

# Save current selection as watchlist
if selected:
    save_as_name = st.text_input("Save Current as Watchlist", key=f"save_watchlist_{mode}", placeholder="e.g., My Portfolio")
    if st.button("Save", key=f"save_watchlist_btn_{mode}"):
        if save_as_name.strip():
            watchlists[save_as_name.strip()] = {
                "tickers": selected.copy(),
                "mode": mode,
                "created": date.today().isoformat()
            }
            save_watchlists(watchlists)
            st.success(f"Saved as '{save_as_name.strip()}'!")
            st.rerun()

# Safety: unique list
selected = list(dict.fromkeys(selected))

# ===== TAB EASY: EASY MODE =====
with tab_easy:
    st.header("🚦 Easy Mode")
    st.markdown("Simple, clear signals without the noise.")
    
    # Ticker selection (reuse existing selection logic or simplified one)
    # Use 'selected' if available (from sidebar), otherwise fallback to universe
    easy_ticker_options = selected if selected else universe
    easy_ticker = st.selectbox("Select Asset", easy_ticker_options, key="easy_ticker_select")
    
    if easy_ticker:
        # Load data
        df_easy = load_data(easy_ticker, start_d, end_d, mode, interval=interval, data_source=data_source)
        
        if not df_easy.empty:
            # Calculate signal
            signal_data = calculate_easy_mode_signal(df_easy)
            
            # Display Traffic Light
            st.markdown(f"""
            <div style="display: flex; justify-content: center; margin: 2rem 0;">
                <div style="
                    width: 200px;
                    height: 200px;
                    border-radius: 50%;
                    background-color: {signal_data['color']};
                    box-shadow: 0 0 50px {signal_data['color']};
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    font-size: 24px;
                    font-weight: bold;
                    color: black;
                    text-align: center;
                ">
                    {signal_data['signal']}
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # ELI5 Explanation
            st.subheader("🧐 What does this mean?")
            for reason in signal_data['reasons']:
                st.info(f"• {reason}")
                
            # Quick Stats
            last_price = df_easy["close"].iloc[-1]
            prev_price = df_easy["close"].iloc[-2]
            change_pct = ((last_price - prev_price) / prev_price) * 100
            
            st.subheader("📊 Quick Stats")
            col1, col2 = st.columns(2)
            col1.metric("Current Price", f"${last_price:,.2f}")
            col2.metric("24h Change", f"{change_pct:+.2f}%", delta_color="normal")
            
            # Suggested Bracket Order
            if "trade_suggestion" in signal_data and signal_data["trade_suggestion"]["side"] != "NEUTRAL":
                st.divider()
                st.subheader("🎯 Suggested Bracket Order")
                s = signal_data["trade_suggestion"]
                
                # Determine colors
                entry_color = "#00d4ff"
                stop_color = "#f31260"
                target_color = "#00ff88"
                
                c1, c2, c3 = st.columns(3)
                with c1:
                    st.markdown(f"**Entry ({s['side']})**")
                    st.markdown(f"<h3 style='color: {entry_color}; margin:0'>${s['entry']:,.2f}</h3>", unsafe_allow_html=True)
                
                with c2:
                    stop_pct = ((s['stop_loss'] - s['entry']) / s['entry']) * 100
                    st.markdown("**Stop Loss**")
                    st.markdown(f"<h3 style='color: {stop_color}; margin:0'>${s['stop_loss']:,.2f}</h3>", unsafe_allow_html=True)
                    st.caption(f"{stop_pct:+.2f}%")
                
                with c3:
                    target_pct = ((s['take_profit'] - s['entry']) / s['entry']) * 100
                    st.markdown("**Take Profit**")
                    st.markdown(f"<h3 style='color: {target_color}; margin:0'>${s['take_profit']:,.2f}</h3>", unsafe_allow_html=True)
                    st.caption(f"{target_pct:+.2f}% (1.5 R/R)")
                
                st.info("💡 This suggestion uses a 2x ATR Stop Loss and 1.5 Risk/Reward Ratio.")

            # Share Card
            st.divider()
            st.subheader("📸 Share Card")
            share_html = generate_share_card_html(easy_ticker, last_price, change_pct, signal_data)
            st.components.v1.html(share_html, height=450)
            st.caption("Take a screenshot of the card above to share!")
            
        else:
            st.warning("No data available for this ticker.")
    else:
        st.info("Select a ticker to see the Easy Mode analysis.")

# ===== TAB 1: TRACKER =====
with tab1:
    # --- Per-ticker charts ---
    for t in selected:
        df = load_data(t, start_d, end_d, mode, interval=interval, data_source=data_source)
        cols = st.columns([5, 1], gap="large")
        with cols[0]:
            # Header with AI Recommendations button
            header_cols = st.columns([3, 1])
            with header_cols[0]:
                st.markdown(f"**{t}**")
            with header_cols[1]:
                # Check if button was clicked and generate recommendation
                ai_recommendation_cache_key = f"ai_rec_cache_{t}"
                
                # Get recommendation for button color (before rendering button)
                button_color = "#00d4ff"  # Default cyan
                button_bg = "linear-gradient(90deg, #00d4ff 0%, #0099cc 100%)"
                if ai_recommendation_cache_key in st.session_state:
                    rec = st.session_state[ai_recommendation_cache_key]
                    if "direction" in rec:
                        if rec["direction"] == "LONG":
                            button_color = "#00ff88"
                            button_bg = "linear-gradient(90deg, #00ff88 0%, #17c964 100%)"
                        elif rec["direction"] == "SHORT":
                            button_color = "#f31260"
                            button_bg = "linear-gradient(90deg, #f31260 0%, #cc0d4d 100%)"
                        elif rec["direction"] == "HOLD":
                            button_color = "#b0b0b0"
                            button_bg = "linear-gradient(90deg, #b0b0b0 0%, #888888 100%)"
                
                # Create a unique marker div to identify this button's location
                button_marker_id = f"ai_rec_marker_{t.replace('-', '_').replace('.', '_')}"
                st.markdown(f'<div id="{button_marker_id}" style="display:none;"></div>', unsafe_allow_html=True)
                
                button_clicked = st.button("🤖 AI Recommendations", key=f"ai_rec_{t}", use_container_width=True)
                
                if button_clicked:
                    st.session_state[f"show_ai_{t}"] = True
                    # Initialize indicator settings if not already set
                    if f"ai_sma_{t}" not in st.session_state:
                        st.session_state[f"ai_sma_{t}"] = True
                    if f"ai_rsi_{t}" not in st.session_state:
                        st.session_state[f"ai_rsi_{t}"] = True
                    if f"ai_macd_{t}" not in st.session_state:
                        st.session_state[f"ai_macd_{t}"] = True
                    if f"ai_supertrend_{t}" not in st.session_state:
                        st.session_state[f"ai_supertrend_{t}"] = True
                    if f"ai_fvg_{t}" not in st.session_state:
                        st.session_state[f"ai_fvg_{t}"] = True
                    if f"ai_news_{t}" not in st.session_state:
                        st.session_state[f"ai_news_{t}"] = True
                    # Get indicator settings from session state
                    use_sma_ai = st.session_state.get(f"ai_sma_{t}", True)
                    use_rsi_ai = st.session_state.get(f"ai_rsi_{t}", True)
                    use_macd_ai = st.session_state.get(f"ai_macd_{t}", True)
                    use_supertrend_ai = st.session_state.get(f"ai_supertrend_{t}", True)
                    use_fvg_ai = st.session_state.get(f"ai_fvg_{t}", True)
                    use_news_ai = st.session_state.get(f"ai_news_{t}", True)
                    # Generate recommendation immediately when clicked
                    with st.spinner(""):
                        # Use same MACD mode and extended settings as chart for AI by default
                        ai_macd_mode = macd_mode
                        recommendation = generate_ai_recommendation(
                            t, df, start_d, end_d, mode, interval,
                            use_sma=use_sma_ai, use_rsi=use_rsi_ai, use_macd=use_macd_ai,
                            use_supertrend=use_supertrend_ai, use_fvg=use_fvg_ai, use_news=use_news_ai,
                            macd_mode=ai_macd_mode,
                            macd_sideways_window=macd_sideways_window,
                            macd_sideways_threshold=macd_sideways_threshold
                        )
                        if "error" not in recommendation:
                            # Store indicator settings with recommendation
                            recommendation["_use_sma"] = use_sma_ai
                            recommendation["_use_rsi"] = use_rsi_ai
                            recommendation["_use_macd"] = use_macd_ai
                            recommendation["_use_supertrend"] = use_supertrend_ai
                            recommendation["_use_fvg"] = use_fvg_ai
                            recommendation["_use_news"] = use_news_ai
                            st.session_state[ai_recommendation_cache_key] = recommendation
                            st.rerun()
                
                # Inject JavaScript AFTER button is rendered to style it
                st.markdown(f"""
                <script>
                (function() {{
                    function styleAIButton() {{
                        // Find the marker div
                        const marker = document.getElementById('{button_marker_id}');
                        if (!marker) return;
                        
                        // Find the button near the marker (should be in the same column)
                        const column = marker.closest('[data-testid="column"]');
                        if (!column) return;
                        
                        // Find button with "AI Recommendations" text in this column
                        const buttons = column.querySelectorAll('button');
                        buttons.forEach(function(btn) {{
                            const btnText = (btn.textContent || btn.innerText || '').trim();
                            if (btnText.includes('AI Recommendations') || btnText.includes('🤖')) {{
                                // Apply styles directly
                                btn.style.setProperty('background', '{button_bg}', 'important');
                                btn.style.setProperty('border-color', '{button_color}', 'important');
                                btn.style.setProperty('background-image', 'none', 'important');
                                btn.style.setProperty('background-color', '{button_color}', 'important');
                                
                                // Override hover state
                                btn.addEventListener('mouseenter', function() {{
                                    this.style.setProperty('background', '{button_bg}', 'important');
                                    this.style.setProperty('opacity', '0.85', 'important');
                                }});
                                btn.addEventListener('mouseleave', function() {{
                                    this.style.setProperty('background', '{button_bg}', 'important');
                                    this.style.setProperty('opacity', '1', 'important');
                                }});
                            }}
                        }});
                    }}
                    
                    // Run multiple times to catch the button
                    if (document.readyState === 'loading') {{
                        document.addEventListener('DOMContentLoaded', styleAIButton);
                    }} else {{
                        styleAIButton();
                    }}
                    setTimeout(styleAIButton, 100);
                    setTimeout(styleAIButton, 300);
                    setTimeout(styleAIButton, 600);
                    
                    // Watch for DOM changes
                    const observer = new MutationObserver(function(mutations) {{
                        styleAIButton();
                    }});
                    observer.observe(document.body, {{
                        childList: true,
                        subtree: true,
                        attributes: false
                    }});
                }})();
                </script>
                """, unsafe_allow_html=True)
            
            # Get trade parameters from AI recommendation if available
            trade_entry = None
            trade_stop_loss = None
            trade_target = None
            trade_direction = None
            if st.session_state.get(f"show_ai_{t}", False):
                ai_rec_cache_key = f"ai_rec_cache_{t}"
                if ai_rec_cache_key in st.session_state:
                    rec = st.session_state[ai_rec_cache_key]
                    if "entry_price" in rec and "stop_loss" in rec and "target_price" in rec and "direction" in rec:
                        trade_entry = rec.get("entry_price")
                        trade_stop_loss = rec.get("stop_loss")
                        trade_target = rec.get("target_price")
                        trade_direction = rec.get("direction")
            
            fig = make_chart(
                df, f"{t} — SMA {SMA_SHORT}/{SMA_LONG}", theme, pretouch,
                show_volume=show_volume, show_rsi=show_rsi,
                show_macd=show_macd, show_bollinger=show_bollinger,
                show_sma20=show_sma20, show_sma200=show_sma200,
                show_ema=show_ema, show_supertrend=show_supertrend,
                show_support_resistance=show_support_resistance,
                show_fvg=show_fvg,
                macd_mode=macd_mode,
                macd_sideways_window=macd_sideways_window,
                macd_sideways_threshold=macd_sideways_threshold,
                trade_entry=trade_entry, trade_stop_loss=trade_stop_loss,
                trade_target=trade_target, trade_direction=trade_direction
            )
            st.plotly_chart(fig, use_container_width=True, key=f"chart_{t}_{mode}")
            
            # Export chart buttons
            export_cols = st.columns(3)
            with export_cols[0]:
                try:
                    img_bytes = fig.to_image(format="png", width=1200, height=800)
                    st.download_button(
                        label="📥 Download PNG",
                        data=img_bytes,
                        file_name=f"{t}_{timeframe}_{date.today().isoformat()}.png",
                        mime="image/png",
                        key=f"export_png_{t}"
                    )
                except Exception as e:
                    st.caption("PNG export requires kaleido. Install: pip install kaleido")
            with export_cols[1]:
                try:
                    html_str = fig.to_html(include_plotlyjs="cdn")
                    st.download_button(
                        label="📥 Download HTML",
                        data=html_str,
                        file_name=f"{t}_{timeframe}_{date.today().isoformat()}.html",
                        mime="text/html",
                        key=f"export_html_{t}"
                    )
                except Exception:
                    pass
            
            # AI Recommendations Report
            if st.session_state.get(f"show_ai_{t}", False):
                # Initialize indicator settings if not already set
                if f"ai_sma_{t}" not in st.session_state:
                    st.session_state[f"ai_sma_{t}"] = True
                if f"ai_rsi_{t}" not in st.session_state:
                    st.session_state[f"ai_rsi_{t}"] = True
                if f"ai_macd_{t}" not in st.session_state:
                    st.session_state[f"ai_macd_{t}"] = True
                if f"ai_supertrend_{t}" not in st.session_state:
                    st.session_state[f"ai_supertrend_{t}"] = True
                if f"ai_fvg_{t}" not in st.session_state:
                    st.session_state[f"ai_fvg_{t}"] = True
                if f"ai_news_{t}" not in st.session_state:
                    st.session_state[f"ai_news_{t}"] = True
                
                # Indicator selection for AI Recommendations
                st.markdown("### 🤖 AI Recommendations Settings")
                st.markdown("**Select indicators to use for AI analysis:**")
                ai_ind_cols = st.columns(3)
                with ai_ind_cols[0]:
                    use_sma_ai = st.checkbox("Use SMA", value=st.session_state.get(f"ai_sma_{t}", True), key=f"ai_sma_{t}")
                    use_rsi_ai = st.checkbox("Use RSI", value=st.session_state.get(f"ai_rsi_{t}", True), key=f"ai_rsi_{t}")
                with ai_ind_cols[1]:
                    use_macd_ai = st.checkbox("Use MACD", value=st.session_state.get(f"ai_macd_{t}", True), key=f"ai_macd_{t}")
                    use_supertrend_ai = st.checkbox("Use Supertrend", value=st.session_state.get(f"ai_supertrend_{t}", True), key=f"ai_supertrend_{t}")
                with ai_ind_cols[2]:
                    use_fvg_ai = st.checkbox("Use Fair Value Gap", value=st.session_state.get(f"ai_fvg_{t}", True), key=f"ai_fvg_{t}")
                    use_news_ai = st.checkbox("Use News Sentiment", value=st.session_state.get(f"ai_news_{t}", True), key=f"ai_news_{t}")
                
                # Get values from session state (checkboxes automatically update session state)
                # We don't need to manually set them - the widgets handle it
                use_sma_ai = st.session_state.get(f"ai_sma_{t}", use_sma_ai)
                use_rsi_ai = st.session_state.get(f"ai_rsi_{t}", use_rsi_ai)
                use_macd_ai = st.session_state.get(f"ai_macd_{t}", use_macd_ai)
                use_supertrend_ai = st.session_state.get(f"ai_supertrend_{t}", use_supertrend_ai)
                use_fvg_ai = st.session_state.get(f"ai_fvg_{t}", use_fvg_ai)
                use_news_ai = st.session_state.get(f"ai_news_{t}", use_news_ai)
                
                st.markdown("---")
                
                # Use cached recommendation if available, but regenerate if indicator settings changed
                ai_recommendation_cache_key = f"ai_rec_cache_{t}"
                # Check if we need to regenerate based on indicator settings
                indicator_settings_changed = False
                if ai_recommendation_cache_key in st.session_state:
                    cached_rec = st.session_state[ai_recommendation_cache_key]
                    # Check if indicator settings match (simple check)
                    if (cached_rec.get("_use_sma") != use_sma_ai or 
                        cached_rec.get("_use_rsi") != use_rsi_ai or
                        cached_rec.get("_use_macd") != use_macd_ai or
                        cached_rec.get("_use_supertrend") != use_supertrend_ai or
                        cached_rec.get("_use_fvg") != use_fvg_ai or
                        cached_rec.get("_use_news") != use_news_ai):
                        indicator_settings_changed = True
                
                if ai_recommendation_cache_key not in st.session_state or indicator_settings_changed or st.button("🔄 Refresh Analysis", key=f"refresh_ai_{t}"):
                    with st.spinner(f"🤖 Analyzing {t}... Generating comprehensive AI recommendation..."):
                        recommendation = generate_ai_recommendation(t, df, start_d, end_d, mode, interval,
                                                                    use_sma=use_sma_ai, use_rsi=use_rsi_ai, use_macd=use_macd_ai,
                                                                    use_supertrend=use_supertrend_ai, use_fvg=use_fvg_ai, use_news=use_news_ai)
                        if "error" not in recommendation:
                            # Store indicator settings with recommendation
                            recommendation["_use_sma"] = use_sma_ai
                            recommendation["_use_rsi"] = use_rsi_ai
                            recommendation["_use_macd"] = use_macd_ai
                            recommendation["_use_supertrend"] = use_supertrend_ai
                            recommendation["_use_fvg"] = use_fvg_ai
                            recommendation["_use_news"] = use_news_ai
                            st.session_state[ai_recommendation_cache_key] = recommendation
                            st.rerun()
                else:
                    recommendation = st.session_state[ai_recommendation_cache_key]
                
                if "error" in recommendation:
                    st.error(f"❌ {recommendation['error']}")
                else:
                    st.markdown("---")
                    st.markdown(f"### 🤖 AI Trading Recommendation for {t}")
                    
                    # Main recommendation card
                    rec_color = "#00ff88" if recommendation["direction"] == "LONG" else "#f31260" if recommendation["direction"] == "SHORT" else "#b0b0b0"
                    rec_bg = "rgba(0, 255, 136, 0.1)" if recommendation["direction"] == "LONG" else "rgba(243, 18, 96, 0.1)" if recommendation["direction"] == "SHORT" else "rgba(176, 176, 176, 0.1)"
                    
                    st.markdown(f"""
                    <div style="background: linear-gradient(135deg, {rec_bg} 0%, rgba(0, 212, 255, 0.1) 100%); 
                                border: 2px solid {rec_color}; border-radius: 12px; padding: 1.5rem; margin: 1rem 0;">
                        <h2 style="color: {rec_color}; margin: 0; text-align: center; font-size: 2.5rem;">
                            {recommendation["recommendation"]}
                        </h2>
                        <p style="color: #b0b0b0; text-align: center; margin: 0.5rem 0 0 0; font-size: 1.1rem;">
                            Confidence: <strong style="color: {rec_color}">{recommendation["confidence"]}</strong> • 
                            Score: <strong style="color: {rec_color}">{recommendation["total_score"]}</strong>
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Analysis breakdown
                    st.markdown("#### 📊 Analysis Breakdown")
                    analysis_cols = st.columns(6)
                    
                    with analysis_cols[0]:
                        st.markdown("**📈 SMA 20/200 Analysis**")
                        sma_status = recommendation["sma_analysis"]["status"]
                        sma_color = "#00ff88" if "Bullish" in sma_status or "Golden" in sma_status else "#f31260" if "Bearish" in sma_status or "Death" in sma_status else "#b0b0b0"
                        st.markdown(f'<p style="color: {sma_color};">{recommendation["sma_analysis"]["description"]}</p>', unsafe_allow_html=True)
                    
                    with analysis_cols[1]:
                        st.markdown("**📉 RSI Analysis**")
                        rsi_status = recommendation["rsi_analysis"]["status"]
                        rsi_color = "#00ff88" if "Oversold" in rsi_status or "Bullish" in rsi_status else "#f31260" if "Overbought" in rsi_status or "Bearish" in rsi_status else "#b0b0b0"
                        st.markdown(f'<p style="color: {rsi_color};">{recommendation["rsi_analysis"]["description"]}</p>', unsafe_allow_html=True)
                    
                    with analysis_cols[2]:
                        st.markdown("**📊 MACD Analysis**")
                        macd_status = recommendation.get("macd_analysis", {}).get("status", "N/A")
                        macd_color = "#00ff88" if "Bullish" in macd_status or "Crossover" in macd_status else "#f31260" if "Bearish" in macd_status else "#b0b0b0"
                        macd_desc = recommendation.get("macd_analysis", {}).get("description", "No MACD data available")
                        st.markdown(f'<p style="color: {macd_color};">{macd_desc}</p>', unsafe_allow_html=True)
                    
                    with analysis_cols[3]:
                        st.markdown("**📊 Supertrend Analysis**")
                        supertrend_status = recommendation.get("supertrend_analysis", {}).get("status", "N/A")
                        supertrend_color = "#00ff88" if "Bullish" in supertrend_status or "Reversal" in supertrend_status else "#f31260" if "Bearish" in supertrend_status else "#b0b0b0"
                        supertrend_desc = recommendation.get("supertrend_analysis", {}).get("description", "No Supertrend data available")
                        st.markdown(f'<p style="color: {supertrend_color};">{supertrend_desc}</p>', unsafe_allow_html=True)
                    
                    with analysis_cols[4]:
                        st.markdown("**⚡ Fair Value Gap Analysis**")
                        fvg_status = recommendation.get("fvg_analysis", {}).get("status", "N/A")
                        fvg_color = "#00ff88" if "Bullish" in fvg_status else "#f31260" if "Bearish" in fvg_status else "#b0b0b0"
                        fvg_desc = recommendation.get("fvg_analysis", {}).get("description", "No FVG data available")
                        st.markdown(f'<p style="color: {fvg_color};">{fvg_desc}</p>', unsafe_allow_html=True)
                    
                    with analysis_cols[5]:
                        st.markdown("**📰 News Sentiment**")
                        news_status = recommendation["news_analysis"]["status"]
                        news_color = "#00ff88" if news_status == "Bullish" else "#f31260" if news_status == "Bearish" else "#b0b0b0"
                        st.markdown(f'<p style="color: {news_color};">{recommendation["news_analysis"]["description"]}</p>', unsafe_allow_html=True)
                        if "confidence" in recommendation["news_analysis"]:
                            st.caption(f"Avg Confidence: {recommendation['news_analysis']['confidence']*100:.0f}%")
                    
                    # Trading parameters
                    st.markdown("---")
                    st.markdown("#### 🎯 Trading Parameters")
                    param_cols = st.columns(4)
                    
                    with param_cols[0]:
                        st.metric("Current Price", f"${recommendation['current_price']:.2f}")
                    with param_cols[1]:
                        st.metric("Entry Price", f"${recommendation['entry_price']:.2f}")
                    with param_cols[2]:
                        st.metric("Stop Loss", f"${recommendation['stop_loss']:.2f}", 
                                 delta=f"-{recommendation['risk_pct']:.1f}%", delta_color="inverse")
                    with param_cols[3]:
                        st.metric("Target Price", f"${recommendation['target_price']:.2f}", 
                                 delta=f"+{recommendation['reward_pct']:.1f}%", delta_color="normal")
                    
                    # Risk/Reward
                    st.markdown("---")
                    rr_cols = st.columns(3)
                    with rr_cols[0]:
                        st.metric("Risk/Reward Ratio", f"{recommendation['risk_reward_ratio']:.2f}:1")
                    with rr_cols[1]:
                        st.metric("Risk", f"{recommendation['risk_pct']:.2f}%")
                    with rr_cols[2]:
                        st.metric("Reward", f"{recommendation['reward_pct']:.2f}%")
                    
                    # Summary
                    st.markdown("---")
                    st.markdown("#### 📝 Summary")
                    supertrend_desc = recommendation.get("supertrend_analysis", {}).get("description", "No Supertrend data available")
                    if recommendation["direction"] == "LONG":
                        summary_text = f"""
                        **Recommendation:** {recommendation["recommendation"]}
                        
                        Based on the analysis:
                        - **SMA Trend:** {recommendation["sma_analysis"]["description"]}
                        - **RSI Condition:** {recommendation["rsi_analysis"]["description"]}
                        - **MACD:** {recommendation.get("macd_analysis", {}).get("description", "No MACD data")}
                        - **Supertrend:** {supertrend_desc}
                        - **Fair Value Gap:** {recommendation.get("fvg_analysis", {}).get("description", "No FVG data")}
                        - **News Sentiment:** {recommendation["news_analysis"]["description"]}
                        
                        **Trading Plan:**
                        - Enter LONG position at ${recommendation['entry_price']:.2f}
                        - Set stop loss at ${recommendation['stop_loss']:.2f} ({recommendation['risk_pct']:.1f}% risk)
                        - Target price: ${recommendation['target_price']:.2f} ({recommendation['reward_pct']:.1f}% reward)
                        - Risk/Reward Ratio: {recommendation['risk_reward_ratio']:.2f}:1
                        """
                    elif recommendation["direction"] == "SHORT":
                        summary_text = f"""
                        **Recommendation:** {recommendation["recommendation"]}
                        
                        Based on the analysis:
                        - **SMA Trend:** {recommendation["sma_analysis"]["description"]}
                        - **RSI Condition:** {recommendation["rsi_analysis"]["description"]}
                        - **MACD:** {recommendation.get("macd_analysis", {}).get("description", "No MACD data")}
                        - **Supertrend:** {supertrend_desc}
                        - **Fair Value Gap:** {recommendation.get("fvg_analysis", {}).get("description", "No FVG data")}
                        - **News Sentiment:** {recommendation["news_analysis"]["description"]}
                        
                        **Trading Plan:**
                        - Enter SHORT position at ${recommendation['entry_price']:.2f}
                        - Set stop loss at ${recommendation['stop_loss']:.2f} ({recommendation['risk_pct']:.1f}% risk)
                        - Target price: ${recommendation['target_price']:.2f} ({recommendation['reward_pct']:.1f}% reward)
                        - Risk/Reward Ratio: {recommendation['risk_reward_ratio']:.2f}:1
                        """
                    else:
                        summary_text = f"""
                        **Recommendation:** {recommendation["recommendation"]}
                        
                        The analysis shows mixed signals. Consider waiting for clearer confirmation before entering a position.
                        
                        - **SMA Trend:** {recommendation["sma_analysis"]["description"]}
                        - **RSI Condition:** {recommendation["rsi_analysis"]["description"]}
                        - **MACD:** {recommendation.get("macd_analysis", {}).get("description", "No MACD data")}
                        - **Supertrend:** {supertrend_desc}
                        - **Fair Value Gap:** {recommendation.get("fvg_analysis", {}).get("description", "No FVG data")}
                        - **News Sentiment:** {recommendation["news_analysis"]["description"]}
                        """
                    
                    st.markdown(summary_text)
                    
                    # Close button
                    if st.button("Close Report", key=f"close_ai_{t}"):
                        st.session_state[f"show_ai_{t}"] = False
                        st.rerun()
        with cols[1]:
            # Container with auto-sizing for metrics card
            st.markdown("""
            <style>
            div[data-testid="column"]:nth-of-type(2) {
                width: fit-content !important;
                min-width: fit-content !important;
                max-width: fit-content !important;
            }
            </style>
            """, unsafe_allow_html=True)
            
            if df.empty:
                st.warning(f"⚠️ No data available for {t}. This could be due to:\n- Yahoo Finance API issues\n- Network connectivity\n- Invalid ticker symbol\n\nTry refreshing the page or check your internet connection.")
            else:
                d = df.copy()
                d["SMA20"] = _sma(d["close"], SMA_SHORT)
                d["SMA200"] = _sma(d["close"], SMA_LONG)
                last = d.iloc[-1]
                price = float(last["close"])
                sma20 = float(last["SMA20"])
                sma200 = float(last["SMA200"])
                st.metric("Last Close", f"${price:,.2f}")
                st.metric(f"SMA {SMA_SHORT}", f"${sma20:,.2f}")
                st.metric(f"SMA {SMA_LONG}", f"${sma200:,.2f}")
                st.metric("20 vs 200", f"{_pct(sma20, sma200):+.2f}%")
                
                # RSI
                if show_rsi:
                    rsi = _rsi(d["close"], window=14)
                    rsi_last_val = rsi.iloc[-1]
                    rsi_last = float(rsi_last_val) if not (pd.isna(rsi_last_val) or math.isnan(rsi_last_val)) else np.nan
                    if not math.isnan(rsi_last):
                        rsi_color = "#f31260" if rsi_last > 70 else "#17c964" if rsi_last < 30 else "#888"
                        st.metric("RSI (14)", f"{rsi_last:.1f}", delta=None)
                
                # MACD
                if show_macd:
                    macd_line, signal_line, histogram = _macd(d["close"])
                    macd_last_val = macd_line.iloc[-1]
                    signal_last_val = signal_line.iloc[-1]
                    hist_last_val = histogram.iloc[-1]
                    macd_last = float(macd_last_val) if not (pd.isna(macd_last_val) or math.isnan(macd_last_val)) else np.nan
                    signal_last = float(signal_last_val) if not (pd.isna(signal_last_val) or math.isnan(signal_last_val)) else np.nan
                    hist_last = float(hist_last_val) if not (pd.isna(hist_last_val) or math.isnan(hist_last_val)) else np.nan
                    if not math.isnan(macd_last):
                        st.metric("MACD", f"{macd_last:.2f}")
                        st.metric("Signal", f"{signal_last:.2f}")
                        st.metric("Histogram", f"{hist_last:.2f}")
                
                # Volume
                if show_volume and not d["volume"].isna().all():
                    vol_last = float(last["volume"])
                    vol_sma = _volume_sma(d["volume"], window=20)
                    vol_sma_last_val = vol_sma.iloc[-1]
                    vol_sma_last = float(vol_sma_last_val) if not (pd.isna(vol_sma_last_val) or math.isnan(vol_sma_last_val)) else np.nan
                    if not math.isnan(vol_sma_last) and vol_sma_last > 0:
                        vol_ratio = vol_last / vol_sma_last
                        st.metric("Volume", f"{vol_last:,.0f}")
                        st.metric("Vol SMA20", f"{vol_sma_last:,.0f}")
                        st.metric("Vol Ratio", f"{vol_ratio:.2f}x")
                
                # Performance Metrics
                metrics = calculate_performance_metrics(df)
                if metrics:
                    st.markdown("---")
                    st.markdown("**📊 Performance Metrics:**")
                    st.metric("Total Return", f"{metrics['total_return']:.2f}%")
                    st.metric("Annualized Return", f"{metrics['annualized_return']:.2f}%")
                    st.metric("Volatility (Annual)", f"{metrics['volatility']:.2f}%")
                    st.metric("Sharpe Ratio", f"{metrics['sharpe_ratio']:.2f}")
                    st.metric("Max Drawdown", f"{metrics['max_drawdown']:.2f}%")
                    st.metric("Win Rate", f"{metrics['win_rate']:.1f}%")
                    st.caption(f"Based on {metrics['days']} {timeframe.lower()} periods")
                
                # Check alerts for this ticker
                alerts = load_alerts()
                triggered = check_alerts(t, df, alerts)
                if triggered:
                    st.markdown("---")
                    st.markdown("**🔔 Alerts:**")
                    for trig in triggered:
                        st.warning(trig["message"])

    # --- Screener ---
    st.divider()
    st.subheader("Pretouch Screener (closest to SMA200 on top)")

    screener_df = build_screener(selected or universe, start_d, end_d, mode, pretouch, interval=interval)
    if screener_df.empty:
        st.info("No data to screen.")
    else:
        # Build column list based on available data
        base_cols = ["Ticker", "Trend", "Touch", "Last", "SMA20", "SMA200", "Dist to SMA200 (%)", "Dist to SMA20 (%)"]
        indicator_cols = []
        if "RSI" in screener_df.columns:
            indicator_cols.append("RSI")
        if "MACD" in screener_df.columns:
            indicator_cols.extend(["MACD", "MACD Signal", "MACD Hist"])
        if "Vol Ratio" in screener_df.columns:
            indicator_cols.extend(["Volume", "Vol Ratio"])
        if "BB Position %" in screener_df.columns:
            indicator_cols.append("BB Position %")
        
        show_cols = base_cols + indicator_cols
        # Filter to only columns that exist
        show_cols = [c for c in show_cols if c in screener_df.columns]
        
        format_dict = {
            "Last": "${:,.2f}",
            "SMA20": "${:,.2f}",
            "SMA200": "${:,.2f}",
            "Dist to SMA200 (%)": "{:+.2f}%",
            "Dist to SMA20 (%)": "{:+.2f}%",
            "RSI": "{:.1f}",
            "MACD": "{:.2f}",
            "MACD Signal": "{:.2f}",
            "MACD Hist": "{:.2f}",
            "Volume": "{:,.0f}",
            "Vol SMA20": "{:,.0f}",
            "Vol Ratio": "{:.2f}x",
            "BB Position %": "{:.1f}%"
        }
        
        s = screener_df[show_cols].style.apply(
            lambda col: [_badge_color(v) for v in col], subset=["Trend"]
        ).format({k: v for k, v in format_dict.items() if k in show_cols})
        st.dataframe(s, use_container_width=True, hide_index=True)
        st.download_button(
            label="Download Screener CSV",
            data=screener_df.to_csv(index=False).encode("utf-8"),
            file_name="pretouch_screener.csv",
            mime="text/csv",
            key="dl_screener_csv"
        )

    # --- Footnote / Diagnostics ---
    with st.expander("About / Diagnostics"):
        st.write("If crypto history appears short, CoinGecko free API caps range to ~365 days. "
                 "The app falls back between Yahoo and CoinGecko automatically.")
        st.write("Built with Streamlit, yfinance, CoinGecko, Plotly. No paid data sources.")

# ===== TAB 2: ALERTS =====
with tab2:
    st.header("🔔 Alerts & Notifications")
    st.markdown("Set up alerts for cross events, price levels, and RSI conditions.")
    
    alerts = load_alerts()
    
    # Display existing alerts
    if alerts:
        st.subheader("Active Alerts")
        for i, alert in enumerate(alerts):
            col1, col2, col3, col4 = st.columns([3, 2, 1, 1])
            with col1:
                alert_type_names = {
                    "golden_cross": "Golden Cross",
                    "death_cross": "Death Cross",
                    "price_above": f"Price Above ${alert.get('value', 0):.2f}",
                    "price_below": f"Price Below ${alert.get('value', 0):.2f}",
                    "rsi_overbought": f"RSI Overbought (≥{alert.get('value', 70)})",
                    "rsi_oversold": f"RSI Oversold (≤{alert.get('value', 30)})"
                }
                st.write(f"**{alert.get('ticker', 'N/A')}** - {alert_type_names.get(alert.get('type', ''), alert.get('type', ''))}")
            with col2:
                status = "✅ Enabled" if alert.get("enabled", True) else "❌ Disabled"
                st.write(status)
            with col3:
                if st.button("Toggle", key=f"toggle_{i}"):
                    alerts[i]["enabled"] = not alerts[i].get("enabled", True)
                    save_alerts(alerts)
                    st.rerun()
            with col4:
                if st.button("Delete", key=f"delete_{i}"):
                    alerts.pop(i)
                    save_alerts(alerts)
                    st.rerun()
        st.divider()
    
    # Add new alert
    st.subheader("Add New Alert")
    alert_col1, alert_col2 = st.columns(2)
    with alert_col1:
        alert_ticker = st.text_input("Ticker", placeholder="e.g., SPY", key="alert_ticker")
        alert_type = st.selectbox("Alert Type", [
            "golden_cross", "death_cross", "price_above", "price_below", 
            "rsi_overbought", "rsi_oversold"
        ], key="alert_type")
    
    with alert_col2:
        alert_value = None
        if alert_type in ["price_above", "price_below"]:
            alert_value = st.number_input("Price Threshold ($)", min_value=0.0, value=0.0, step=0.01, key="alert_price")
        elif alert_type in ["rsi_overbought", "rsi_oversold"]:
            alert_value = st.number_input("RSI Threshold", min_value=0.0, max_value=100.0, 
                                         value=70.0 if alert_type == "rsi_overbought" else 30.0, 
                                         step=1.0, key="alert_rsi")
    
    if st.button("Add Alert", key="add_alert"):
        if alert_ticker.strip():
            new_alert = {
                "ticker": alert_ticker.strip().upper(),
                "type": alert_type,
                "enabled": True,
                "created": datetime.now().isoformat()
            }
            if alert_value is not None:
                new_alert["value"] = float(alert_value)
            alerts.append(new_alert)
            save_alerts(alerts)
            st.success(f"Alert added for {new_alert['ticker']}")
            st.rerun()
    
    # Email Configuration
    st.divider()
    st.subheader("📧 Email Configuration")
    email_config = load_email_config()
    
    with st.expander("Configure Email Alerts"):
        email_from = st.text_input("From Email", value=email_config.get("from_email", ""), 
                                   placeholder="your.email@gmail.com", key="email_from")
        email_password = st.text_input("Password/App Password", type="password", 
                                      value=email_config.get("password", ""), key="email_password")
        email_to = st.text_input("Default To Email", value=email_config.get("default_email", ""), 
                                placeholder="recipient@example.com", key="email_to")
        smtp_server = st.text_input("SMTP Server", value=email_config.get("smtp_server", "smtp.gmail.com"), key="smtp_server")
        smtp_port = st.number_input("SMTP Port", value=email_config.get("smtp_port", 587), min_value=1, max_value=65535, key="smtp_port")
        
        if st.button("Save Email Config", key="save_email_config"):
            email_config = {
                "from_email": email_from,
                "password": email_password,
                "default_email": email_to,
                "smtp_server": smtp_server,
                "smtp_port": int(smtp_port)
            }
            save_email_config(email_config)
            st.success("Email configuration saved!")
            st.info("💡 For Gmail, use an App Password instead of your regular password. Enable 2FA and generate an app password.")
    
    # Alert History
    st.divider()
    st.subheader("📜 Alert History")
    alert_history = load_alert_history()
    if alert_history:
        recent_history = alert_history[-20:]  # Show last 20
        history_data = []
        for h in reversed(recent_history):
            history_data.append({
                "Time": h.get("timestamp", "")[:19] if len(h.get("timestamp", "")) > 19 else h.get("timestamp", ""),
                "Ticker": h.get("ticker", ""),
                "Type": h.get("type", ""),
                "Message": h.get("message", ""),
                "Email Sent": "✅" if h.get("email_sent") else "❌"
            })
        history_df = pd.DataFrame(history_data)
        st.dataframe(history_df, use_container_width=True, hide_index=True)
    else:
        st.info("No alert history yet.")

# ===== TAB 3: CROSS HISTORY =====
with tab3:
    st.header("📊 Historical Cross Analysis")
    st.markdown("Analyze past Golden and Death Cross events and their performance.")
    
    hist_ticker = st.selectbox("Select Ticker", selected or universe, key="hist_ticker")
    days_after = st.slider("Analyze performance after (days)", min_value=7, max_value=180, value=30, step=7, key="hist_days")
    
    if st.button("Analyze Crosses", key="analyze_crosses"):
        df = load_data(hist_ticker, start_d, end_d, mode, interval=interval, data_source=data_source)
        if not df.empty:
            df = df.copy()
            df["SMA20"] = _sma(df["close"], SMA_SHORT)
            df["SMA200"] = _sma(df["close"], SMA_LONG)
            
            crosses = detect_crosses(df)
            crosses_with_perf = analyze_cross_performance(df, crosses, days_after)
            
            # Save to history
            history = load_cross_history()
            history[hist_ticker] = crosses_with_perf
            save_cross_history(history)
            
            if crosses_with_perf:
                st.subheader(f"Cross Events for {hist_ticker}")
                
                # Summary statistics
                golden_crosses = [c for c in crosses_with_perf if c["type"] == "Golden"]
                death_crosses = [c for c in crosses_with_perf if c["type"] == "Death"]
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Crosses", len(crosses_with_perf))
                with col2:
                    st.metric("Golden Crosses", len(golden_crosses))
                with col3:
                    st.metric("Death Crosses", len(death_crosses))
                with col4:
                    if golden_crosses:
                        avg_perf = np.mean([c["performance"]["pct_change"] for c in golden_crosses if c.get("performance")])
                        st.metric("Avg Golden Return", f"{avg_perf:.2f}%")
                
                # Display table
                display_data = []
                for cross in crosses_with_perf:
                    row = {
                        "Date": cross["date"],
                        "Type": cross["type"],
                        "Price at Cross": f"${cross['price']:.2f}",
                        "SMA20": f"${cross['sma20']:.2f}",
                        "SMA200": f"${cross['sma200']:.2f}"
                    }
                    if cross.get("performance"):
                        perf = cross["performance"]
                        row["Price After"] = f"${perf['price_after']:.2f}"
                        row["Return %"] = f"{perf['pct_change']:+.2f}%"
                        row["Days"] = f"{perf['days_actual']}"
                    else:
                        row["Price After"] = "N/A"
                        row["Return %"] = "N/A"
                        row["Days"] = "N/A"
                    display_data.append(row)
                
                df_display = pd.DataFrame(display_data)
                st.dataframe(df_display, use_container_width=True, hide_index=True)
            else:
                st.info(f"No cross events found for {hist_ticker} in the selected period.")
        else:
            st.error(f"No data available for {hist_ticker}")

# ===== TAB 4: PORTFOLIO =====
with tab4:
    st.header("💼 Portfolio Tracking")
    st.markdown("Track multiple tickers as a portfolio with custom weights.")
    
    portfolio = load_portfolio()
    
    # Portfolio tickers
    st.subheader("Portfolio Tickers")
    available_tickers = selected or universe
    saved_tickers = portfolio.get("tickers", [])
    # Filter saved tickers to only include those in available options
    valid_defaults = [t for t in saved_tickers if t in available_tickers]
    
    portfolio_tickers = st.multiselect(
        "Select tickers for portfolio",
        options=available_tickers,
        default=valid_defaults,
        key="portfolio_tickers"
    )
    
    if portfolio_tickers:
        st.subheader("Portfolio Weights")
        st.caption("Weights will be normalized to sum to 100%. Leave empty for equal weighting.")
        
        weights = {}
        for ticker in portfolio_tickers:
            current_weight = portfolio.get("weights", {}).get(ticker, 0.0)
            weight = st.number_input(
                f"{ticker} Weight (%)",
                min_value=0.0,
                max_value=100.0,
                value=current_weight if current_weight > 0 else (100.0 / len(portfolio_tickers)),
                step=1.0,
                key=f"weight_{ticker}"
            )
            weights[ticker] = weight
        
        if st.button("Update Portfolio", key="update_portfolio"):
            portfolio = {"tickers": portfolio_tickers, "weights": weights}
            save_portfolio(portfolio)
            st.success("Portfolio updated!")
            st.rerun()
        
        # Calculate and display portfolio metrics
        if st.button("Calculate Portfolio Metrics", key="calc_portfolio"):
            metrics = calculate_portfolio_metrics(portfolio, start_d, end_d, mode)
            
            if "error" not in metrics:
                st.subheader("Portfolio Performance")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Portfolio Return", f"{metrics['portfolio_return']:.2f}%")
                with col2:
                    st.metric("Number of Tickers", metrics['ticker_count'])
                with col3:
                    st.metric("Portfolio Value", f"${metrics['portfolio_value']:,.2f}")
                
                st.subheader("Individual Performance")
                perf_data = []
                for ticker, data in metrics['individual_returns'].items():
                    perf_data.append({
                        "Ticker": ticker,
                        "Weight": f"{data['weight']*100:.1f}%",
                        "Return": f"{data['return']:.2f}%",
                        "Price": f"${data['price']:,.2f}"
                    })
                perf_df = pd.DataFrame(perf_data)
                st.dataframe(perf_df, use_container_width=True, hide_index=True)
            else:
                st.error(metrics["error"])

# ===== TAB 5: BACKTESTING =====
with tab5:
    st.header("🧪 Strategy Backtesting")
    st.markdown("Test trading strategies based on SMA crosses and compare performance.")
    
    backtest_ticker = st.selectbox("Select Ticker for Backtest", selected or universe, key="backtest_ticker")
    strategy = st.selectbox("Strategy", ["golden_death", "buy_hold"], key="backtest_strategy")
    initial_capital = st.number_input("Initial Capital ($)", min_value=1000.0, value=10000.0, step=1000.0, key="backtest_capital")
    
    strategy_names = {
        "golden_death": "Golden/Death Cross Strategy (Buy on Golden Cross, Sell on Death Cross)",
        "buy_hold": "Buy and Hold Strategy"
    }
    st.caption(f"**Strategy:** {strategy_names.get(strategy, strategy)}")
    
    if st.button("Run Backtest", key="run_backtest"):
        df = load_data(backtest_ticker, start_d, end_d, mode, interval=interval, data_source=data_source)
        if not df.empty:
            result = backtest_strategy(df, strategy, initial_capital)
            
            if "error" not in result:
                st.subheader("Backtest Results")
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Initial Capital", f"${result['initial_capital']:,.2f}")
                with col2:
                    st.metric("Final Capital", f"${result['final_capital']:,.2f}")
                with col3:
                    st.metric("Total Return", f"{result['total_return']:.2f}%")
                with col4:
                    st.metric("Max Drawdown", f"{result['max_drawdown']:.2f}%")
                
                st.metric("Number of Trades", result['num_trades'])
                
                # Trade history
                if result['trades']:
                    st.subheader("Trade History")
                    trades_data = []
                    for trade in result['trades']:
                        trade_date = trade['date']
                        if isinstance(trade_date, pd.Timestamp):
                            trade_date = trade_date.strftime("%Y-%m-%d")
                        trades_data.append({
                            "Date": trade_date,
                            "Type": trade['type'],
                            "Price": f"${trade['price']:.2f}",
                            "Shares": f"{trade.get('shares', 0):.4f}" if 'shares' in trade else "N/A",
                            "Capital": f"${trade.get('capital', 0):,.2f}" if 'capital' in trade else "N/A"
                        })
                    trades_df = pd.DataFrame(trades_data)
                    st.dataframe(trades_df, use_container_width=True, hide_index=True)
            else:
                st.error(result["error"])
        else:
            st.error(f"No data available for {backtest_ticker}")

# ===== TAB 6: NEWS =====
with tab6:
    st.header("📰 News & Sentiment")
    news_ticker = st.selectbox("Select Ticker for News", selected or universe, key="news_ticker")
    
    if st.button("Fetch News", key="fetch_news"):
        with st.spinner(f"Fetching news for {news_ticker}..."):
            news_items = fetch_news(news_ticker, max_items=10)
        
        if news_items:
            st.success(f"Found {len(news_items)} news articles")
            
            # Calculate individual sentiment counts
            bullish_count = sum(1 for item in news_items if item.get('sentiment', {}).get('sentiment') == 'BULLISH')
            bearish_count = sum(1 for item in news_items if item.get('sentiment', {}).get('sentiment') == 'BEARISH')
            neutral_count = len(news_items) - bullish_count - bearish_count
            
            # Calculate overall combined sentiment
            total_bullish_score = sum(item.get('sentiment', {}).get('score', 0) for item in news_items if item.get('sentiment', {}).get('sentiment') == 'BULLISH')
            total_bearish_score = sum(abs(item.get('sentiment', {}).get('score', 0)) for item in news_items if item.get('sentiment', {}).get('sentiment') == 'BEARISH')
            total_confidence = sum(item.get('sentiment', {}).get('confidence', 0) for item in news_items) / len(news_items) if news_items else 0
            
            # Determine overall sentiment
            if bullish_count > bearish_count and bullish_count > neutral_count:
                overall_sentiment = "🟢 BULLISH"
                overall_color = "green"
            elif bearish_count > bullish_count and bearish_count > neutral_count:
                overall_sentiment = "🔴 BEARISH"
                overall_color = "red"
            else:
                overall_sentiment = "⚪ NEUTRAL"
                overall_color = "gray"
            
            # Overall sentiment indicator (prominent)
            st.markdown("---")
            st.markdown(f"### 📊 Overall News Sentiment for {news_ticker}")
            overall_col1, overall_col2, overall_col3 = st.columns([2, 1, 1])
            with overall_col1:
                st.markdown(f"<h2 style='color:{overall_color};text-align:center;'>{overall_sentiment}</h2>", unsafe_allow_html=True)
            with overall_col2:
                sentiment_score_pct = ((bullish_count - bearish_count) / len(news_items) * 100) if news_items else 0
                delta_label = "Positive" if bullish_count > bearish_count else "Negative" if bearish_count > bullish_count else "Neutral"
                st.metric("Sentiment Score", f"{sentiment_score_pct:.0f}%", delta=delta_label)
            with overall_col3:
                st.metric("Confidence", f"{total_confidence * 100:.0f}%")
            
            # Individual breakdown
            st.markdown("---")
            st.markdown("### 📈 Sentiment Breakdown")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Articles", len(news_items))
            with col2:
                st.metric("🟢 Bullish", bullish_count, delta=f"{bullish_count}/{len(news_items)}")
            with col3:
                st.metric("🔴 Bearish", bearish_count, delta=f"{bearish_count}/{len(news_items)}")
            with col4:
                st.metric("⚪ Neutral", neutral_count, delta=f"{neutral_count}/{len(news_items)}")
            
            st.divider()
            
            # Sort articles by sentiment: Bullish first, then Bearish, then Neutral
            sentiment_order = {'BULLISH': 0, 'BEARISH': 1, 'NEUTRAL': 2}
            sorted_news = sorted(news_items, key=lambda x: (
                sentiment_order.get(x.get('sentiment', {}).get('sentiment', 'NEUTRAL'), 2),
                -x.get('sentiment', {}).get('confidence', 0)  # Higher confidence first within same sentiment
            ))
            
            # Display each article
            for i, item in enumerate(sorted_news, 1):
                # Handle different news item formats
                title = item.get('title') or item.get('headline') or f"Article {i}"
                publisher = item.get('publisher') or item.get('source') or item.get('provider') or "Unknown"
                
                # Get sentiment
                sentiment_data = item.get('sentiment', {})
                sentiment = sentiment_data.get('sentiment', 'NEUTRAL')
                confidence = sentiment_data.get('confidence', 0.0)
                
                # Color code by sentiment
                if sentiment == 'BULLISH':
                    sentiment_emoji = "🟢"
                    sentiment_color = "green"
                elif sentiment == 'BEARISH':
                    sentiment_emoji = "🔴"
                    sentiment_color = "red"
                else:
                    sentiment_emoji = "⚪"
                    sentiment_color = "gray"
                
                # Format publish time
                pub_time = item.get('providerPublishTime') or item.get('pubDate') or item.get('publishedAt')
                if pub_time:
                    try:
                        if isinstance(pub_time, (int, float)):
                            from datetime import datetime
                            pub_time = datetime.fromtimestamp(pub_time).strftime("%Y-%m-%d %H:%M:%S")
                        elif isinstance(pub_time, str) and len(pub_time) > 10:
                            pub_time = pub_time[:19]  # Truncate if too long
                    except Exception:
                        pass
                
                # Create expander with sentiment badge
                expander_title = f"{sentiment_emoji} **{title}** - {publisher}"
                
                with st.expander(expander_title, expanded=True):  # Expand all articles by default
                    # Sentiment badge
                    sentiment_col1, sentiment_col2 = st.columns([3, 1])
                    with sentiment_col1:
                        st.markdown(f"**Sentiment:** <span style='color:{sentiment_color};font-weight:bold'>{sentiment_emoji} {sentiment}</span> (Confidence: {confidence*100:.0f}%)", unsafe_allow_html=True)
                    with sentiment_col2:
                        link = item.get('link') or item.get('url')
                        if link:
                            st.markdown(f"[🔗 Read Full Article]({link})", unsafe_allow_html=True)
                        else:
                            st.caption("No link available")
                    
                    if pub_time:
                        st.write(f"**Published:** {pub_time}")
                    
                    # Article summary/description - try multiple fields
                    summary = item.get('summary') or item.get('description') or item.get('text') or item.get('content')
                    if summary and len(summary.strip()) > 10:  # Only show if meaningful
                        st.write("**Summary:**")
                        st.write(summary)
                    else:
                        # If no summary, try to get more info from the raw item
                        raw_text = str(item.get('rawText', '')) or str(item.get('text', ''))
                        if raw_text and len(raw_text.strip()) > 10:
                            st.write("**Content:**")
                            st.write(raw_text[:500] + "..." if len(raw_text) > 500 else raw_text)
                        else:
                            st.info("💡 No summary available in the API response. Click the link above to read the full article on Yahoo Finance.")
                    
                    # Show sentiment details
                    if sentiment_data.get('bullish_keywords', 0) > 0 or sentiment_data.get('bearish_keywords', 0) > 0:
                        st.caption(f"📊 Analysis: {sentiment_data.get('bullish_keywords', 0)} bullish keywords, {sentiment_data.get('bearish_keywords', 0)} bearish keywords")
                    elif title and len(title) > 5:  # If we only have title, analyze it
                        title_sentiment = analyze_sentiment(title)
                        if title_sentiment.get('bullish_keywords', 0) > 0 or title_sentiment.get('bearish_keywords', 0) > 0:
                            st.caption(f"📊 Title Analysis: {title_sentiment.get('bullish_keywords', 0)} bullish, {title_sentiment.get('bearish_keywords', 0)} bearish keywords")
                    
                    # Debug: Show raw data structure (collapsed)
                    if st.checkbox(f"🔍 Show Raw Data", key=f"debug_{i}_{news_ticker}", value=False):
                        st.json(item)
        else:
            st.warning(f"❌ No news found for {news_ticker}")
            st.write("**Possible reasons:**")
            st.write("- Yahoo Finance API may not have news for this ticker")
            st.write("- Network connectivity issues")
            st.write("- Ticker symbol format issue (try without ^ for indices)")
            
            # Try alternative ticker format
            st.write("\n**💡 Suggestions:**")
            st.write(f"- Try: **{news_ticker.replace('^', '')}** (without ^)")
            st.write("- Try major tickers: **AAPL**, **TSLA**, **MSFT**, **SPY**")
            st.write("- Check the browser console (F12) for error messages")
            
            # Debug info
            with st.expander("🔍 Debug Information", expanded=True):
                st.write("**Testing news fetch...**")
                try:
                    yf = _yf()
                    test_stock = yf.Ticker(news_ticker)
                    st.write(f"✅ Ticker object created for: {news_ticker}")
                    st.write(f"Has 'news' attribute: {hasattr(test_stock, 'news')}")
                    
                    if hasattr(test_stock, 'news'):
                        try:
                            test_news = test_stock.news
                            st.write(f"**News type:** {type(test_news)}")
                            
                            if test_news is None:
                                st.error("❌ News is None")
                            elif isinstance(test_news, list):
                                st.write(f"**News list length:** {len(test_news)}")
                                if len(test_news) > 0:
                                    st.success(f"✅ Found {len(test_news)} news items!")
                                    st.write(f"**First item type:** {type(test_news[0])}")
                                    
                                    if isinstance(test_news[0], dict):
                                        keys = list(test_news[0].keys())
                                        st.write(f"**First item keys ({len(keys)} total):** {', '.join(keys[:20])}")
                                        if len(keys) > 20:
                                            st.caption(f"... and {len(keys) - 20} more keys")
                                        
                                        # Show sample of first item
                                        st.write("**First item sample:**")
                                        sample = {k: str(v)[:100] for k, v in list(test_news[0].items())[:10]}
                                        st.json(sample)
                                    else:
                                        st.write(f"**First item:** {str(test_news[0])[:200]}")
                                else:
                                    st.warning("⚠️ News list is empty")
                            else:
                                st.write(f"**News is not a list, it's:** {type(test_news)}")
                                st.write(f"**Value:** {str(test_news)[:200]}")
                        except Exception as e:
                            st.error(f"❌ Error accessing news: {e}")
                            import traceback
                            st.code(traceback.format_exc())
                    
                    # Also try alternative tickers
                    st.write("\n**💡 Try these test tickers:**")
                    test_tickers = ["AAPL", "TSLA", "MSFT", "SPY"]
                    for test_t in test_tickers:
                        if st.button(f"Test {test_t}", key=f"test_{test_t}"):
                            try:
                                test_stock2 = yf.Ticker(test_t)
                                test_news2 = test_stock2.news
                                if test_news2 and len(test_news2) > 0:
                                    st.success(f"✅ {test_t} has {len(test_news2)} news items!")
                                else:
                                    st.warning(f"⚠️ {test_t} has no news")
                            except Exception as e2:
                                st.error(f"❌ Error with {test_t}: {e2}")
                except Exception as e:
                    st.error(f"❌ Error creating ticker: {e}")
                    import traceback
                    st.code(traceback.format_exc())

# ===== TAB 7: PATTERNS =====
with tab7:
    st.header("📐 Chart Pattern Detection")
    pattern_ticker = st.selectbox("Select Ticker for Pattern Analysis", selected or universe, key="pattern_ticker")
    
    if st.button("Analyze Patterns", key="analyze_patterns"):
        df = load_data(pattern_ticker, start_d, end_d, mode, interval=interval, data_source=data_source)
        if not df.empty:
            patterns = detect_patterns(df)
            if patterns:
                st.subheader(f"Detected Patterns for {pattern_ticker}")
                
                # Determine if pattern is bullish or bearish
                def get_pattern_sentiment(pattern_type: str) -> tuple[str, str]:
                    """Returns (emoji, color) for pattern type."""
                    bullish_patterns = ['Double Bottom', 'Inverse Head and Shoulders', 'Ascending Triangle', 'Uptrend']
                    bearish_patterns = ['Double Top', 'Head and Shoulders', 'Descending Triangle', 'Downtrend']
                    
                    if pattern_type in bullish_patterns:
                        return ("🟢", "green")
                    elif pattern_type in bearish_patterns:
                        return ("🔴", "red")
                    else:
                        return ("⚪", "gray")
                
                for pattern in patterns:
                    pattern_type = pattern['type']
                    emoji, color = get_pattern_sentiment(pattern_type)
                    
                    col1, col2 = st.columns([2, 1])
                    with col1:
                        st.markdown(f"{emoji} **<span style='color:{color}'>{pattern_type}</span>**", unsafe_allow_html=True)
                        st.caption(pattern.get('description', ''))
                    with col2:
                        confidence_pct = pattern.get('confidence', 0) * 100
                        st.metric("Confidence", f"{confidence_pct:.0f}%")
            else:
                st.info(f"No patterns detected for {pattern_ticker}")
        else:
            st.error(f"No data available for {pattern_ticker}")

# ===== TAB 8: CORRELATION =====
with tab8:
    st.header("🔗 Correlation Analysis")
    st.markdown("Analyze correlation between multiple tickers.")
    
    if len(selected) >= 2:
        correlation_df = calculate_correlation(selected, start_d, end_d, mode, interval=interval)
        if not correlation_df.empty:
            st.subheader("Correlation Matrix")
            # Create heatmap
            fig_corr = go.Figure(data=go.Heatmap(
                z=correlation_df.values,
                x=correlation_df.columns,
                y=correlation_df.index,
                colorscale='RdBu',
                zmid=0,
                text=correlation_df.values,
                texttemplate='%{text:.2f}',
                textfont={"size": 10},
                colorbar=dict(title="Correlation")
            ))
            fig_corr.update_layout(
                title="Correlation Matrix",
                width=800,
                height=600
            )
            st.plotly_chart(fig_corr, use_container_width=True)
            
            st.subheader("Correlation Data")
            # Format correlation values for display
            formatted_df = correlation_df.round(3)
            st.dataframe(formatted_df, use_container_width=True)
        else:
            st.warning("Could not calculate correlation. Ensure all tickers have data.")
    else:
        st.info("Select at least 2 tickers to analyze correlation.")

# ===== TAB 9: TRADE JOURNAL =====
with tab9:
    st.header("📝 Trade Journal")
    st.markdown("Log your trades and track performance.")
    
    journal = load_trade_journal()
    
    # Add new trade
    st.subheader("Add New Trade")
    with st.form("add_trade_form"):
        col1, col2 = st.columns(2)
        with col1:
            trade_ticker = st.text_input("Ticker", key="trade_ticker")
            trade_type = st.selectbox("Type", ["Buy", "Sell"], key="trade_type")
            trade_price = st.number_input("Price", min_value=0.0, step=0.01, key="trade_price")
        with col2:
            trade_shares = st.number_input("Shares", min_value=0.0, step=0.01, key="trade_shares")
            trade_date = st.date_input("Date", value=date.today(), key="trade_date")
            trade_notes = st.text_area("Notes", key="trade_notes")
        
        if st.form_submit_button("Add Trade"):
            if trade_ticker and trade_price > 0 and trade_shares > 0:
                new_trade = {
                    "ticker": trade_ticker.upper(),
                    "type": trade_type,
                    "price": float(trade_price),
                    "shares": float(trade_shares),
                    "date": trade_date.isoformat(),
                    "notes": trade_notes,
                    "timestamp": datetime.now().isoformat()
                }
                journal.append(new_trade)
                save_trade_journal(journal)
                st.success("Trade added!")
                st.rerun()
    
    # Display trades
    if journal:
        st.subheader("Trade History")
        journal_df = pd.DataFrame(journal)
        journal_df = journal_df.sort_values("date", ascending=False)
        
        # Calculate P&L if we have current prices
        if st.checkbox("Calculate P&L", key="calc_pnl"):
            for idx, trade in journal_df.iterrows():
                ticker = trade["ticker"]
                df_current = load_data(ticker, start_d, end_d, mode, interval=interval, data_source=data_source)
                if not df_current.empty:
                    current_price = float(df_current["close"].iloc[-1])
                    if trade["type"] == "Buy":
                        pnl = (current_price - trade["price"]) * trade["shares"]
                        pnl_pct = ((current_price - trade["price"]) / trade["price"]) * 100
                    else:
                        pnl = (trade["price"] - current_price) * trade["shares"]
                        pnl_pct = ((trade["price"] - current_price) / current_price) * 100
                    journal_df.at[idx, "Current Price"] = current_price
                    journal_df.at[idx, "P&L"] = pnl
                    journal_df.at[idx, "P&L %"] = pnl_pct
        
        st.dataframe(journal_df, use_container_width=True, hide_index=True)
        
        # Delete trade section
        st.divider()
        st.subheader("🗑️ Delete Trade")
        if len(journal) > 0:
            # Create a list of trade identifiers for selection
            trade_options = []
            for idx, trade in enumerate(journal):
                trade_date = trade.get("date", "Unknown")
                trade_ticker = trade.get("ticker", "Unknown")
                trade_type = trade.get("type", "Unknown")
                trade_price = trade.get("price", 0)
                trade_shares = trade.get("shares", 0)
                # Create a readable identifier
                trade_label = f"{idx + 1}. {trade_date} - {trade_ticker} - {trade_type} - ${trade_price:.2f} x {trade_shares}"
                trade_options.append((idx, trade_label))
            
            if trade_options:
                selected_trade_idx = st.selectbox(
                    "Select trade to delete:",
                    options=[i for i, _ in trade_options],
                    format_func=lambda x: trade_options[x][1],
                    key="delete_trade_select"
                )
                
                col1, col2 = st.columns([1, 4])
                with col1:
                    if st.button("🗑️ Delete Selected Trade", key="delete_trade_btn", type="primary"):
                        if selected_trade_idx is not None and 0 <= selected_trade_idx < len(journal):
                            deleted_trade = journal.pop(selected_trade_idx)
                            save_trade_journal(journal)
                            st.success(f"✅ Trade deleted: {deleted_trade.get('ticker', 'Unknown')} - {deleted_trade.get('date', 'Unknown')}")
                            st.rerun()
                        else:
                            st.error("Invalid trade selection")
        
        # Summary
        if len(journal) > 0:
            st.subheader("Summary")
            total_trades = len(journal)
            buy_trades = len([t for t in journal if t.get("type") == "Buy"])
            sell_trades = len([t for t in journal if t.get("type") == "Sell"])
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Trades", total_trades)
            with col2:
                st.metric("Buy Trades", buy_trades)
            with col3:
                st.metric("Sell Trades", sell_trades)
    else:
        st.info("No trades logged yet. Add your first trade above.")
    
    # Win Rate Analytics
    st.divider()
    st.subheader("📊 Strategy Performance Analytics")
    
    if journal:
        analytics_start = st.date_input("Analysis Start Date", value=start_d, key="analytics_start")
        analytics_end = st.date_input("Analysis End Date", value=end_d, key="analytics_end")
        
        if st.button("Analyze Performance", key="analyze_performance"):
            performance = analyze_strategy_performance(journal, analytics_start, analytics_end, mode)
            
            if performance:
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Trades", performance.get("total_trades", 0))
                with col2:
                    st.metric("Completed Trades", performance.get("completed_trades", 0))
                with col3:
                    win_rate = performance.get("win_rate", 0)
                    st.metric("Win Rate", f"{win_rate:.1f}%", 
                             delta=f"{win_rate - 50:.1f}%" if win_rate > 50 else None)
                with col4:
                    avg_return = performance.get("avg_return", 0)
                    st.metric("Avg Return", f"{avg_return:.2f}%",
                             delta=f"{avg_return:.2f}%" if avg_return > 0 else None)
                
                col5, col6 = st.columns(2)
                with col5:
                    avg_win = performance.get("avg_win", 0)
                    st.metric("Avg Win", f"{avg_win:.2f}%")
                with col6:
                    avg_loss = performance.get("avg_loss", 0)
                    st.metric("Avg Loss", f"{avg_loss:.2f}%")
                
                # Best and worst trades
                if performance.get("best_trade"):
                    best = performance["best_trade"]
                    st.success(f"🏆 Best Trade: {best['ticker']} - {best['pnl_pct']:.2f}% return")
                
                if performance.get("worst_trade"):
                    worst = performance["worst_trade"]
                    st.error(f"⚠️ Worst Trade: {worst['ticker']} - {worst['pnl_pct']:.2f}% return")
                
                # Performance by ticker
                ticker_perf = performance.get("ticker_performance", {})
                if ticker_perf:
                    st.subheader("Performance by Ticker")
                    ticker_data = []
                    for ticker, stats in ticker_perf.items():
                        win_rate_ticker = (stats["wins"] / stats["trades"] * 100) if stats["trades"] > 0 else 0
                        ticker_data.append({
                            "Ticker": ticker,
                            "Trades": stats["trades"],
                            "Wins": stats["wins"],
                            "Win Rate": f"{win_rate_ticker:.1f}%",
                            "Total Return": f"{stats['total_pnl']:.2f}%"
                        })
                    ticker_df = pd.DataFrame(ticker_data)
                    st.dataframe(ticker_df, use_container_width=True, hide_index=True)
            else:
                st.info("No completed trades in the selected date range.")

# ===== TAB 10: SIGNAL GENERATOR =====
with tab10:
    st.header("⚡ Trading Signal Generator")
    st.markdown("Get AI-powered buy/sell signals by combining multiple indicators.")
    
    signal_ticker = st.selectbox("Select Ticker", selected or universe, key="signal_ticker")
    
    if st.button("Generate Signal", key="generate_signal"):
        df = load_data(signal_ticker, start_d, end_d, mode, interval=interval, data_source=data_source)
        if not df.empty:
            signal = generate_trading_signal(df)
            
            # Display signal
            signal_type = signal.get("signal", "NEUTRAL")
            strength = signal.get("strength", 0)
            score = signal.get("score", 0)
            
            # Color code signal
            if "BUY" in signal_type:
                signal_color = "🟢"
                signal_emoji = "📈"
            elif "SELL" in signal_type:
                signal_color = "🔴"
                signal_emoji = "📉"
            else:
                signal_color = "🟡"
                signal_emoji = "➡️"
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Signal", f"{signal_emoji} {signal_type}", delta=f"Score: {score}")
            with col2:
                st.metric("Strength", f"{abs(strength)}/10", 
                         delta="Strong" if abs(strength) >= 6 else "Moderate" if abs(strength) >= 3 else "Weak")
            with col3:
                st.metric("Current Price", f"${signal.get('current_price', 0):.2f}")
            
            # Signal details
            st.subheader("Signal Details")
            details = signal.get("details", {})
            detail_cols = st.columns(2)
            with detail_cols[0]:
                st.write("**SMA Trend:**", details.get("sma_trend", "N/A"))
                if "sma_cross" in details:
                    st.success(f"**Cross:** {details['sma_cross']}")
            with detail_cols[1]:
                st.write("**RSI:**", details.get("rsi", "N/A"))
                st.write("**MACD:**", details.get("macd", "N/A"))
                st.write("**Volume:**", details.get("volume", "N/A"))
            
            # Entry/Exit recommendations
            st.subheader("Entry/Exit Recommendations")
            rec_cols = st.columns(3)
            with rec_cols[0]:
                st.metric("Entry Price", f"${signal.get('entry_price', 0):.2f}")
            with rec_cols[1]:
                st.metric("Stop Loss", f"${signal.get('stop_loss', 0):.2f}")
            with rec_cols[2]:
                st.metric("Target Price", f"${signal.get('target_price', 0):.2f}")
            
            # Risk/Reward
            rr = signal.get("risk_reward", {})
            if rr:
                st.subheader("Risk/Reward Analysis")
                rr_cols = st.columns(4)
                with rr_cols[0]:
                    st.metric("Risk/Reward", f"{rr.get('risk_reward_ratio', 0):.2f}:1")
                with rr_cols[1]:
                    st.metric("Risk", f"${rr.get('risk', 0):.2f} ({rr.get('risk_pct', 0):.2f}%)")
                with rr_cols[2]:
                    st.metric("Reward", f"${rr.get('reward', 0):.2f} ({rr.get('reward_pct', 0):.2f}%)")
                with rr_cols[3]:
                    if rr.get('risk_reward_ratio', 0) >= 2:
                        st.success("✅ Good R:R")
                    elif rr.get('risk_reward_ratio', 0) >= 1:
                        st.warning("⚠️ Acceptable R:R")
                    else:
                        st.error("❌ Poor R:R")
        else:
            st.error(f"No data available for {signal_ticker}")

# ===== TAB 11: RISK MANAGEMENT =====
with tab11:
    st.header("🛡️ Risk Management Dashboard")
    st.markdown("Calculate position sizes, stop-losses, and risk/reward ratios.")
    
    # Position Size Calculator
    st.subheader("Position Size Calculator")
    
    # Symbol selector
    risk_ticker = st.selectbox("Select Symbol", selected or universe, key="risk_ticker")
    
    # Fetch current price if symbol is selected
    current_price_risk = None
    if risk_ticker:
        df_risk = load_data(risk_ticker, start_d, end_d, mode, interval=interval, data_source=data_source)
        if not df_risk.empty:
            current_price_risk = float(df_risk["close"].iloc[-1])
            st.info(f"📊 Current {risk_ticker} price: ${current_price_risk:.2f}")
    
    risk_col1, risk_col2 = st.columns(2)
    with risk_col1:
        account_size = st.number_input("Account Size ($)", min_value=0.0, value=10000.0, step=1000.0, key="account_size")
        risk_percent = st.slider("Risk Per Trade (%)", min_value=0.1, max_value=10.0, value=2.0, step=0.1, key="risk_percent")
    with risk_col2:
        entry_price = st.number_input("Entry Price ($)", min_value=0.0, value=current_price_risk if current_price_risk else 100.0, step=0.01, key="entry_price")
        stop_loss_price = st.number_input("Stop Loss ($)", min_value=0.0, value=current_price_risk * 0.95 if current_price_risk else 95.0, step=0.01, key="stop_loss_price")
    
    if st.button("Calculate Position Size", key="calc_position"):
        position = calculate_position_size(account_size, risk_percent, entry_price, stop_loss_price)
        
        if position["shares"] > 0:
            pos_cols = st.columns(3)
            with pos_cols[0]:
                st.metric("Shares to Buy", f"{position['shares']:.2f}")
            with pos_cols[1]:
                st.metric("Position Value", f"${position['position_value']:,.2f}")
            with pos_cols[2]:
                st.metric("Dollar Risk", f"${position['dollar_risk']:,.2f}")
            
            st.info(f"💡 Risk per share: ${position['risk_per_share']:.2f}")
        else:
            st.error("Invalid inputs. Check entry price and stop loss.")
    
    # ATR-based Stop Loss Calculator
    st.divider()
    st.subheader("ATR-Based Stop Loss Calculator")
    atr_ticker = st.selectbox("Select Ticker for ATR Stop", selected or universe, key="atr_ticker")
    atr_multiplier = st.slider("ATR Multiplier", min_value=1.0, max_value=5.0, value=2.0, step=0.5, key="atr_mult")
    is_long_position = st.radio("Position Type", ["Long", "Short"], key="position_type") == "Long"
    
    if st.button("Calculate ATR Stop Loss", key="calc_atr_stop"):
        df = load_data(atr_ticker, start_d, end_d, mode, interval=interval, data_source=data_source)
        if not df.empty:
            atr_stop = calculate_atr_stop_loss(df, atr_multiplier, is_long_position)
            current_price = float(df["close"].iloc[-1])
            
            atr_cols = st.columns(2)
            with atr_cols[0]:
                st.metric("Current Price", f"${current_price:.2f}")
            with atr_cols[1]:
                st.metric("ATR Stop Loss", f"${atr_stop:.2f}")
            
            stop_distance = abs(current_price - atr_stop)
            stop_pct = (stop_distance / current_price) * 100
            st.info(f"Stop loss is ${stop_distance:.2f} ({stop_pct:.2f}%) away from current price")
        else:
            st.error(f"No data available for {atr_ticker}")
    
    # Risk/Reward Calculator
    st.divider()
    st.subheader("Risk/Reward Ratio Calculator")
    
    # Symbol selector
    rr_ticker = st.selectbox("Select Symbol", selected or universe, key="rr_ticker")
    
    # Fetch current price if symbol is selected
    current_price_rr = None
    if rr_ticker:
        df_rr = load_data(rr_ticker, start_d, end_d, mode, interval=interval, data_source=data_source)
        if not df_rr.empty:
            current_price_rr = float(df_rr["close"].iloc[-1])
            st.info(f"📊 Current {rr_ticker} price: ${current_price_rr:.2f}")
    
    rr_col1, rr_col2, rr_col3 = st.columns(3)
    with rr_col1:
        rr_entry = st.number_input("Entry Price", min_value=0.0, value=current_price_rr if current_price_rr else 100.0, step=0.01, key="rr_entry")
    with rr_col2:
        rr_stop = st.number_input("Stop Loss", min_value=0.0, value=current_price_rr * 0.95 if current_price_rr else 95.0, step=0.01, key="rr_stop")
    with rr_col3:
        rr_target = st.number_input("Target Price", min_value=0.0, value=current_price_rr * 1.10 if current_price_rr else 110.0, step=0.01, key="rr_target")
    
    if st.button("Calculate Risk/Reward", key="calc_rr"):
        rr_result = calculate_risk_reward(rr_entry, rr_stop, rr_target)
        
        if rr_result["risk_reward_ratio"] > 0:
            rr_display_cols = st.columns(4)
            with rr_display_cols[0]:
                st.metric("Risk/Reward", f"{rr_result['risk_reward_ratio']:.2f}:1")
            with rr_display_cols[1]:
                st.metric("Risk", f"${rr_result['risk']:.2f} ({rr_result['risk_pct']:.2f}%)")
            with rr_display_cols[2]:
                st.metric("Reward", f"${rr_result['reward']:.2f} ({rr_result['reward_pct']:.2f}%)")
            with rr_display_cols[3]:
                if rr_result['risk_reward_ratio'] >= 3:
                    st.success("✅ Excellent")
                elif rr_result['risk_reward_ratio'] >= 2:
                    st.success("✅ Good")
                elif rr_result['risk_reward_ratio'] >= 1:
                    st.warning("⚠️ Acceptable")
                else:
                    st.error("❌ Poor")
        else:
            st.error("Invalid inputs")

# ===== TAB 12: KRAKEN TRADING =====
with tab12:
    st.header("🔷 Kraken Trading")
    st.markdown("Place and manage trades on Kraken exchange using API integration.")
    st.warning("⚠️ **WARNING**: This feature allows you to place REAL trades with REAL money. Use with extreme caution!")
    
    # API Configuration
    st.subheader("🔐 API Configuration")
    kraken_config = load_kraken_config()
    
    with st.expander("Configure Kraken API Credentials", expanded=not kraken_config.get("api_key")):
        st.info("💡 Get your API keys from: https://www.kraken.com/u/security/api")
        st.info("🔒 API keys are stored securely in Firebase Firestore (encrypted and user-specific)")
        
        api_key = st.text_input(
            "API Key",
            value=kraken_config.get("api_key", ""),
            type="password",
            key="kraken_api_key",
            help="Your Kraken API key"
        )
        api_secret = st.text_input(
            "API Secret",
            value=kraken_config.get("api_secret", ""),
            type="password",
            key="kraken_api_secret",
            help="Your Kraken API secret (private key)"
        )
        
        if st.button("Save API Credentials", key="save_kraken_config"):
            if api_key and api_secret:
                save_kraken_config({"api_key": api_key, "api_secret": api_secret})
                st.success("✅ API credentials saved!")
                st.rerun()
            else:
                st.error("Please enter both API key and secret")
    
    # Check SDK installation
    try:
        from kraken.spot import SpotClient, User, Trade
        st.success("✅ Kraken SDK is installed")
    except ImportError:
        try:
            import krakenex
            st.success("✅ Kraken SDK (krakenex) is installed")
        except ImportError:
            st.error("❌ Kraken SDK not found. Please run: `pip install python-kraken-sdk`")
    
    # Initialize client
    client = get_kraken_client()
    
    if client:
        st.success("✅ Kraken API connected")
        
        # Balance Section
        st.divider()
        st.subheader("💰 Account Balance")
        if st.button("Refresh Balance", key="refresh_balance"):
            with st.spinner("Fetching balance..."):
                balance_result = get_kraken_balance(client)
                if "error" in balance_result:
                    st.error(f"Error: {balance_result['error']}")
                else:
                    if "balance" in balance_result:
                        balance_data = balance_result["balance"]
                        # Display balances in a nice format
                        if isinstance(balance_data, dict):
                            # Filter out zero balances and show non-zero assets
                            non_zero_balances = {k: v for k, v in balance_data.items() if float(v) > 0.0001}
                            if non_zero_balances:
                                cols = st.columns(min(4, len(non_zero_balances)))
                                for idx, (asset, amount) in enumerate(non_zero_balances.items()):
                                    with cols[idx % 4]:
                                        st.metric(asset, f"{float(amount):.6f}")
                            else:
                                st.info("No non-zero balances found")
        
        # Open Orders Section
        st.divider()
        st.subheader("📋 Open Orders")
        if st.button("Refresh Open Orders", key="refresh_orders"):
            with st.spinner("Fetching open orders..."):
                orders_result = get_kraken_open_orders(client)
                if "error" in orders_result:
                    st.error(f"Error: {orders_result['error']}")
                else:
                    if "orders" in orders_result:
                        orders_data = orders_result["orders"]
                        if orders_data and len(orders_data) > 0:
                            st.dataframe(orders_data, use_container_width=True)
                        else:
                            st.info("No open orders")
        
        # Place Order Section
        st.divider()
        st.subheader("📊 Place New Order")
        
        # Get current ticker from sidebar if available
        current_ticker = ""
        if "selected" in st.session_state and st.session_state.get("selected"):
            selected_list = st.session_state["selected"]
            if isinstance(selected_list, list) and len(selected_list) > 0:
                current_ticker = selected_list[0]
            elif isinstance(selected_list, dict):
                mode_val = st.session_state.get("saved_mode", "Stocks")
                if mode_val in selected_list and len(selected_list[mode_val]) > 0:
                    current_ticker = selected_list[mode_val][0]
        
        order_col1, order_col2 = st.columns(2)
        with order_col1:
            # Try to convert current ticker to Kraken pair
            kraken_pair_input = st.text_input(
                "Trading Pair",
                value=convert_ticker_to_kraken_pair(current_ticker) if current_ticker else "XBTUSD",
                placeholder="e.g., XBTUSD, ETHUSD",
                key="kraken_pair",
                help="Kraken trading pair (e.g., XBTUSD for Bitcoin, ETHUSD for Ethereum)"
            )
            
            order_type = st.selectbox(
                "Order Type",
                ["buy", "sell"],
                key="kraken_order_type"
            )
            
            order_mode = st.selectbox(
                "Order Mode",
                ["market", "limit"],
                key="kraken_order_mode",
                help="Market: Execute immediately at current price. Limit: Set your desired price."
            )
        
        with order_col2:
            volume = st.number_input(
                "Volume",
                min_value=0.0,
                value=0.01,
                step=0.001,
                format="%.6f",
                key="kraken_volume",
                help="Amount of base currency to buy/sell"
            )
            
            limit_price = None
            if order_mode == "limit":
                limit_price = st.number_input(
                    "Limit Price",
                    min_value=0.0,
                    value=0.0,
                    step=0.01,
                    format="%.2f",
                    key="kraken_limit_price",
                    help="Price at which to execute the limit order"
                )
            
            dry_run_mode = st.checkbox(
                "🔒 Dry Run Mode (Validate Only)",
                value=True,
                key="kraken_dry_run",
                help="When enabled, validates order parameters without placing actual trade"
            )
        
        # Safety warning
        if not dry_run_mode:
            st.error("⚠️ **REAL TRADE MODE**: This will place an ACTUAL trade with REAL money!")
            confirm_trade = st.checkbox(
                "I understand this is a real trade and I accept the risks",
                key="confirm_real_trade"
            )
        else:
            confirm_trade = True
            st.info("🔒 Dry run mode: Order will be validated but not executed")
        
        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            if st.button("📤 Place Order", key="place_kraken_order", type="primary", disabled=not confirm_trade):
                if not kraken_pair_input:
                    st.error("Please enter a trading pair")
                elif volume <= 0:
                    st.error("Volume must be greater than 0")
                elif order_mode == "limit" and (limit_price is None or limit_price <= 0):
                    st.error("Limit price required for limit orders")
                else:
                    with st.spinner("Placing order..."):
                        result = place_kraken_order(
                            client,
                            pair=kraken_pair_input.upper(),
                            type=order_type,
                            ordertype=order_mode,
                            volume=volume,
                            price=limit_price,
                            dry_run=dry_run_mode
                        )
                        
                        if "error" in result:
                            st.error(f"❌ Error: {result['error']}")
                        else:
                            if dry_run_mode:
                                st.success(f"✅ {result.get('message', 'Order parameters validated')}")
                            else:
                                st.success("✅ Order placed successfully!")
                                if "order" in result:
                                    st.json(result["order"])
        
        with col_btn2:
            if st.button("🔄 Test Connection", key="test_kraken_connection"):
                with st.spinner("Testing connection..."):
                    test_result = get_kraken_balance(client)
                    if "error" in test_result:
                        st.error(f"❌ Connection failed: {test_result['error']}")
                    else:
                        st.success("✅ Connection successful!")
        
        # Bracket Order Section
        st.divider()
        st.subheader("📊 Bracket Order (Entry + Stop-Loss + Take-Profit)")
        st.markdown("Place a complete trade setup with automatic risk management.")
        
        bracket_col1, bracket_col2 = st.columns(2)
        
        with bracket_col1:
            # Try to convert current ticker to Kraken pair
            bracket_pair = st.text_input(
                "Trading Pair",
                value=convert_ticker_to_kraken_pair(current_ticker) if current_ticker else "XBTUSD",
                placeholder="e.g., XBTUSD, ETHUSD",
                key="bracket_pair",
                help="Kraken trading pair"
            )
            
            bracket_side = st.selectbox(
                "Side",
                ["buy", "sell"],
                key="bracket_side"
            )
            
            bracket_volume = st.number_input(
                "Volume",
                min_value=0.0,
                value=0.001,
                step=0.001,
                format="%.6f",
                key="bracket_volume",
                help="Amount to trade"
            )
        
        with bracket_col2:
            bracket_entry_type = st.radio(
                "Entry Type",
                ["Market", "Limit"],
                horizontal=True,
                key="bracket_entry_type",
                help="Market = immediate execution, Limit = execute at specific price"
            )
            
            bracket_entry_price = None
            if bracket_entry_type == "Limit":
                bracket_entry_price = st.number_input(
                    "Entry Price",
                    min_value=0.0,
                    value=50000.0,
                    step=0.01,
                    format="%.2f",
                    key="bracket_entry_price",
                    help="Price at which to enter the trade"
                )
            else:
                st.info("Market order will execute at current market price")
                # For market orders, we'll use a placeholder for validation display
                bracket_entry_price = st.number_input(
                    "Estimated Entry Price (for R/R calc)",
                    min_value=0.0,
                    value=50000.0,
                    step=0.01,
                    format="%.2f",
                    key="bracket_entry_price_est",
                    help="Estimated entry price for risk/reward calculation"
                )
        
        # Stop-loss and Take-profit inputs
        st.markdown("#### Risk Management Levels")
        risk_col1, risk_col2 = st.columns(2)
        
        with risk_col1:
            bracket_stop_loss = st.number_input(
                "🛑 Stop-Loss Price",
                min_value=0.0,
                value=48000.0 if bracket_side == "buy" else 52000.0,
                step=0.01,
                format="%.2f",
                key="bracket_stop_loss",
                help="Price at which to exit if trade goes against you"
            )
        
        with risk_col2:
            bracket_take_profit = st.number_input(
                "🎯 Take-Profit Price",
                min_value=0.0,
                value=55000.0 if bracket_side == "buy" else 45000.0,
                step=0.01,
                format="%.2f",
                key="bracket_take_profit",
                help="Price at which to take profits"
            )
        
        # Validate prices and show feedback
        if bracket_entry_price and bracket_stop_loss and bracket_take_profit:
            validation = validate_bracket_prices(bracket_side, bracket_entry_price, bracket_stop_loss, bracket_take_profit)
            
            if validation["valid"]:
                st.success("✅ Price levels are valid")
                
                # Calculate and display risk/reward metrics
                metrics = calculate_bracket_metrics(bracket_entry_price, bracket_stop_loss, bracket_take_profit, bracket_volume)
                
                st.markdown("#### 📊 Risk/Reward Analysis")
                metric_cols = st.columns(4)
                
                with metric_cols[0]:
                    st.metric(
                        "Risk",
                        f"${metrics['total_risk']:.2f}",
                        f"{metrics['risk_pct']:.2f}%",
                        delta_color="inverse"
                    )
                
                with metric_cols[1]:
                    st.metric(
                        "Reward",
                        f"${metrics['total_reward']:.2f}",
                        f"{metrics['reward_pct']:.2f}%"
                    )
                
                with metric_cols[2]:
                    rr_ratio = metrics['risk_reward_ratio']
                    st.metric(
                        "R/R Ratio",
                        f"{rr_ratio:.2f}:1"
                    )
                
                with metric_cols[3]:
                    if rr_ratio >= 3:
                        st.success("✅ Excellent")
                    elif rr_ratio >= 2:
                        st.success("✅ Good")
                    elif rr_ratio >= 1:
                        st.warning("⚠️ Acceptable")
                    else:
                        st.error("❌ Poor")
                
                # Visual price level indicator
                with st.expander("📈 Price Level Visualization"):
                    if bracket_side == "buy":
                        st.markdown(f"""
                        ```
                        🎯 Take-Profit: ${bracket_take_profit:,.2f}  (+{metrics['reward_pct']:.2f}%)
                                    ↑
                        📍 Entry:       ${bracket_entry_price:,.2f}
                                    ↓
                        🛑 Stop-Loss:   ${bracket_stop_loss:,.2f}  (-{metrics['risk_pct']:.2f}%)
                        ```
                        """)
                    else:
                        st.markdown(f"""
                        ```
                        🛑 Stop-Loss:   ${bracket_stop_loss:,.2f}  (+{metrics['risk_pct']:.2f}%)
                                    ↑
                        📍 Entry:       ${bracket_entry_price:,.2f}
                                    ↓
                        🎯 Take-Profit: ${bracket_take_profit:,.2f}  (-{metrics['reward_pct']:.2f}%)
                        ```
                        """)
            else:
                st.error(f"❌ {validation['error']}")
        
        # Dry run and execution controls
        st.divider()
        bracket_dry_run = st.checkbox(
            "🔒 Dry Run Mode (Validate Only)",
            value=True,
            key="bracket_dry_run",
            help="When enabled, validates order parameters without placing actual trades"
        )
        
        # Safety warning
        if not bracket_dry_run:
            st.error("⚠️ **REAL TRADE MODE**: This will place THREE ACTUAL trades with REAL money!")
            bracket_confirm = st.checkbox(
                "I understand this will place 3 real orders (entry, stop-loss, take-profit) and I accept the risks",
                key="bracket_confirm_trade"
            )
        else:
            bracket_confirm = True
            st.info("🔒 Dry run mode: Orders will be validated but not executed")
        
        # Place bracket order button
        bracket_btn_col1, bracket_btn_col2 = st.columns(2)
        with bracket_btn_col1:
            if st.button("📤 Place Bracket Order", key="place_bracket_order", type="primary", disabled=not bracket_confirm):
                if not bracket_pair:
                    st.error("Please enter a trading pair")
                elif bracket_volume <= 0:
                    st.error("Volume must be greater than 0")
                elif bracket_entry_type == "Limit" and (not bracket_entry_price or bracket_entry_price <= 0):
                    st.error("Entry price required for limit orders")
                elif not bracket_stop_loss or bracket_stop_loss <= 0:
                    st.error("Stop-loss price required")
                elif not bracket_take_profit or bracket_take_profit <= 0:
                    st.error("Take-profit price required")
                else:
                    # Use None for market orders
                    final_entry_price = bracket_entry_price if bracket_entry_type == "Limit" else None
                    
                    with st.spinner("Placing bracket order..."):
                        bracket_result = place_bracket_order(
                            client,
                            pair=bracket_pair.upper(),
                            side=bracket_side,
                            volume=bracket_volume,
                            entry_price=final_entry_price,
                            stop_loss_price=bracket_stop_loss,
                            take_profit_price=bracket_take_profit,
                            dry_run=bracket_dry_run
                        )
                        
                        if "error" in bracket_result:
                            st.error(f"❌ Error: {bracket_result['error']}")
                        else:
                            if bracket_dry_run:
                                st.success(f"✅ {bracket_result.get('message', 'Bracket order validated')}")
                                st.info(f"Entry: {bracket_result.get('entry_type', 'N/A')} | Stop-Loss: {bracket_result.get('stop_loss_type', 'N/A')} | Take-Profit: {bracket_result.get('take_profit_type', 'N/A')}")
                            else:
                                # Real order placed
                                if bracket_result.get("success"):
                                    st.success(f"✅ {bracket_result['message']}")
                                    st.balloons() # 🎉 Victory Confetti!
                                    
                                    # Show order details
                                    if bracket_result.get("entry_order"):
                                        with st.expander("📝 Entry Order Details"):
                                            st.json(bracket_result["entry_order"])
                                    
                                    if bracket_result.get("stop_loss_order"):
                                        with st.expander("🛑 Stop-Loss Order Details"):
                                            st.json(bracket_result["stop_loss_order"])
                                    
                                    if bracket_result.get("take_profit_order"):
                                        with st.expander("🎯 Take-Profit Order Details"):
                                            st.json(bracket_result["take_profit_order"])
                                    
                                    # Show any errors
                                    if bracket_result.get("errors"):
                                        st.warning("⚠️ Some orders had issues:")
                                        for err in bracket_result["errors"]:
                                            st.error(err)
                                else:
                                    st.error(f"❌ {bracket_result.get('message', 'Bracket order failed')}")
                                    if bracket_result.get("errors"):
                                        for err in bracket_result["errors"]:
                                            st.error(err)
        
        with bracket_btn_col2:
            if st.button("🔄 Reset Form", key="reset_bracket_form"):
                st.rerun()
        
        # Order Management Section
        st.divider()
        st.subheader("🗑️ Cancel Orders")
        cancel_txid = st.text_input(
            "Order Transaction ID (TXID)",
            placeholder="Enter TXID to cancel",
            key="cancel_txid"
        )
        if st.button("Cancel Order", key="cancel_kraken_order"):
            if not cancel_txid:
                st.error("Please enter a transaction ID")
            else:
                with st.spinner("Cancelling order..."):
                    cancel_result = cancel_kraken_order(client, cancel_txid)
                    if "error" in cancel_result:
                        st.error(f"❌ Error: {cancel_result['error']}")
                    else:
                        st.success("✅ Order cancelled successfully!")
                        st.json(cancel_result.get("result", {}))
        
        # Help Section
        st.divider()
        with st.expander("ℹ️ Help & Information"):
            st.markdown("""
            **Trading Pair Format:**
            - Kraken uses specific pair formats (e.g., XBTUSD for Bitcoin, ETHUSD for Ethereum)
            - Common pairs: XBTUSD (Bitcoin), ETHUSD (Ethereum), ADAUSD (Cardano), SOLUSD (Solana)
            
            **Order Types:**
            - **Market**: Executes immediately at current market price
            - **Limit**: Executes only when price reaches your specified limit price
            
            **Safety Features:**
            - Dry run mode validates orders without executing
            - Confirmation required for real trades
            - All API credentials stored securely in Firebase Firestore
            
            **Important Notes:**
            - Always test with small amounts first
            - Double-check all order parameters before placing real trades
            - Kraken may charge trading fees
            - Order execution depends on market conditions and liquidity
            """)
            
    else:
        st.info("👆 Please configure your Kraken API credentials above to start trading")
        st.markdown("""
        **Getting Started:**
        1. Log in to your Kraken account
        2. Go to Security > API
        3. Create a new API key with trading permissions
        4. Copy the API Key and Private Key (API Secret)
        5. Enter them in the configuration section above
        6. Enable dry run mode to test without placing real trades
        """)

