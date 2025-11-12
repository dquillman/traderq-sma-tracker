import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="SMA 20/200 Cross Demo", layout="wide")

ticker = st.text_input("Ticker", "SPY").strip().upper()
if not ticker:
    st.stop()

# --- download & prep data
data = yf.download(ticker, period="2y", interval="1d", progress=False).dropna()
if data.empty:
    st.error("No data returned.")
    st.stop()

price_col = "Adj Close" if "Adj Close" in data.columns else "Close"
data["SMA20"] = data[price_col].rolling(20).mean()
data["SMA200"] = data[price_col].rolling(200).mean()

# --- detect crosses
s_prev = data["SMA20"].shift(1)
l_prev = data["SMA200"].shift(1)
cross_up = (s_prev <= l_prev) & (data["SMA20"] > data["SMA200"])
cross_dn = (s_prev >= l_prev) & (data["SMA20"] < data["SMA200"])

# --- chart
fig = go.Figure()
fig.add_trace(go.Scatter(x=data.index, y=data[price_col],
                         mode="lines", name="Price", line=dict(color="#a0a0a0", width=1)))
fig.add_trace(go.Scatter(x=data.index, y=data["SMA20"],
                         mode="lines", name="SMA 20", line=dict(color="#00ff99", width=2)))
fig.add_trace(go.Scatter(x=data.index, y=data["SMA200"],
                         mode="lines", name="SMA 200", line=dict(color="#ff3366", width=2)))

# --- add markers (make them large & obvious)
if cross_up.any():
    fig.add_trace(go.Scatter(
        x=data.index[cross_up], y=data.loc[cross_up, price_col],
        mode="markers", name="Golden Cross",
        marker=dict(symbol="triangle-up", size=16, color="#00ff99",
                    line=dict(width=2, color="#003322")),
        hovertemplate="Golden Cross<br>%{x|%b %d %Y}<br>Price: %{y:.2f}<extra></extra>"
    ))
if cross_dn.any():
    fig.add_trace(go.Scatter(
        x=data.index[cross_dn], y=data.loc[cross_dn, price_col],
        mode="markers", name="Death Cross",
        marker=dict(symbol="triangle-down", size=16, color="#ff3366",
                    line=dict(width=2, color="#330011")),
        hovertemplate="Death Cross<br>%{x|%b %d %Y}<br>Price: %{y:.2f}<extra></extra>"
    ))

golden = int(cross_up.sum())
death = int(cross_dn.sum())
st.caption(f"Crosses — golden {golden}, death {death}")

fig.update_layout(
    template="plotly_dark",
    height=700,
    xaxis=dict(showgrid=False),
    yaxis=dict(showgrid=True, gridcolor="#222"),
    legend=dict(bgcolor="rgba(0,0,0,0.3)"),
)
st.plotly_chart(fig, use_container_width=True)

