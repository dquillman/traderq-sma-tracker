"""
Charts module for TraderQ
Handles Plotly chart generation with technical indicators, FVG, support/resistance, and more
"""

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Import indicator functions - these will be passed as parameters to avoid circular imports
# from modules.indicators import (
#     _sma, _ema, _rsi, _macd, _macd_extended, _bollinger_bands,
#     _volume_sma, _supertrend, find_fair_value_gaps, find_support_resistance
# )

# Constants
SMA_SHORT = 20
SMA_LONG = 200


def add_cross_markers(fig: go.Figure, df: pd.DataFrame,
                      price_col: str = "close",
                      s20: str = "SMA20",
                      s200: str = "SMA200",
                      row: int = 1, col: int = 1) -> None:
    """
    Add Golden Cross and Death Cross markers to the chart
    Golden cross: 20 crosses above 200 (green triangle-up)
    Death cross: 20 crosses below 200 (red triangle-down)
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


def make_chart(df: pd.DataFrame, title: str, theme: str, pretouch_pct: float | None,
               show_volume: bool = True, show_rsi: bool = True, show_macd: bool = True,
               show_bollinger: bool = True, show_sma20: bool = True, show_sma200: bool = True,
               show_ema: bool = False, show_supertrend: bool = False,
               show_support_resistance: bool = False, show_fvg: bool = False,
               macd_mode: str = "Extended",
               macd_sideways_window: int = 10,
               macd_sideways_threshold: float = 8.0,
               trade_entry: float | None = None, trade_stop_loss: float | None = None,
               trade_target: float | None = None, trade_direction: str | None = None,
               _sma=None, _ema=None, _rsi=None, _macd=None, _macd_extended=None,
               _bollinger_bands=None, _volume_sma=None, _supertrend=None,
               find_fair_value_gaps=None, find_support_resistance=None) -> go.Figure:
    """
    Create comprehensive Plotly chart with candlesticks and technical indicators

    NOTE: Indicator functions must be passed as parameters to avoid circular imports.
    Pass functions from modules.indicators module.
    """
    # Ensure timezone-naive index to avoid UTC offset issues
    if not df.empty:
        df = df.copy()
        if isinstance(df.index, pd.DatetimeIndex) and df.index.tz is not None:
            df.index = df.index.tz_localize(None)
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
                font_size=20,
                font_family="Arial, sans-serif",
                font_color="#212529" if theme == "Light" else "white"
            )
        )
        return fig

    df = df.copy()
    if isinstance(df.index, pd.DatetimeIndex) and df.index.tz is not None:
        df.index = df.index.tz_localize(None)

    # Calculate SMAs if functions provided
    if _sma:
        df["SMA20"] = _sma(df["close"], SMA_SHORT)
        df["SMA200"] = _sma(df["close"], SMA_LONG)

    # Set theme colors
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

    # Determine number of subplots
    num_subplots = 1  # Main price chart
    if show_volume:
        num_subplots += 1
    if show_rsi:
        num_subplots += 1
    if show_macd:
        num_subplots += 1

    # Create subplots
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

    # Add filled area between SMAs (must be before candlestick)
    if show_sma20 and show_sma200 and "SMA20" in df.columns and "SMA200" in df.columns:
        sma20_above = df["SMA20"] > df["SMA200"]
        transitions = (sma20_above != sma20_above.shift(1)).fillna(False)
        transition_indices = df.index[transitions].tolist()
        all_points = [df.index[0]] + transition_indices + [df.index[-1]]

        # Ensure timezone-naive timestamps
        for i, idx in enumerate(all_points):
            if isinstance(idx, pd.Timestamp) and idx.tz is not None:
                all_points[i] = idx.tz_localize(None)

        # Create segments with different fill colors
        for i in range(len(all_points) - 1):
            start_idx = all_points[i]
            end_idx = all_points[i + 1]

            try:
                start_mask = df.index >= start_idx
                end_mask = df.index <= end_idx
                segment_mask = start_mask & end_mask

                if segment_mask.any():
                    segment_df = df.loc[segment_mask]
                else:
                    continue
            except Exception:
                continue

            if len(segment_df) < 2:
                continue

            is_sma20_top = segment_df["SMA20"].iloc[0] > segment_df["SMA200"].iloc[0]
            upper = segment_df[["SMA20", "SMA200"]].max(axis=1)
            lower = segment_df[["SMA20", "SMA200"]].min(axis=1)

            if is_sma20_top:
                fillcolor = "rgba(40, 167, 69, 0.15)" if theme == "Light" else "rgba(23, 201, 100, 0.2)"
            else:
                fillcolor = "rgba(220, 53, 69, 0.15)" if theme == "Light" else "rgba(243, 18, 96, 0.2)"

            fig.add_trace(go.Scatter(
                x=segment_df.index, y=upper, mode="lines",
                line=dict(width=0), showlegend=False, hoverinfo="skip"
            ), row=row, col=1)

            fig.add_trace(go.Scatter(
                x=segment_df.index, y=lower, mode="lines",
                line=dict(width=0), fill="tonexty",
                fillcolor=fillcolor, showlegend=False, hoverinfo="skip"
            ), row=row, col=1)

    # Main candlestick chart
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df["open"], high=df["high"], low=df["low"], close=df["close"],
        name="Price"
    ), row=row, col=1)

    # Add SMA lines
    if show_sma20 and "SMA20" in df.columns:
        sma20_color = "#0066cc" if theme == "Light" else "#4fa3ff"
        fig.add_trace(go.Scatter(
            x=df.index, y=df["SMA20"], mode="lines", name=f"SMA {SMA_SHORT}",
            line=dict(width=2, color=sma20_color)
        ), row=row, col=1)

    if show_sma200 and "SMA200" in df.columns:
        sma200_color = "#cc0000" if theme == "Light" else "#ff6b6b"
        fig.add_trace(go.Scatter(
            x=df.index, y=df["SMA200"], mode="lines", name=f"SMA {SMA_LONG}",
            line=dict(width=2, color=sma200_color)
        ), row=row, col=1)

    # Add EMA if enabled
    if show_ema and _ema:
        ema20 = _ema(df["close"], window=20)
        ema_color = "#00a86b" if theme == "Light" else "#00ff88"
        fig.add_trace(go.Scatter(
            x=df.index, y=ema20, mode="lines", name="EMA 20",
            line=dict(width=2, color=ema_color, dash="dot")
        ), row=row, col=1)

    # Add Supertrend if enabled
    if show_supertrend and _supertrend:
        supertrend, trend = _supertrend(df, period=10, multiplier=3.0)
        uptrend_color = "#28a745" if theme == "Light" else "#00ff88"
        downtrend_color = "#dc3545" if theme == "Light" else "#ff4444"

        trend_changes = (trend != trend.shift(1)).fillna(False)
        change_indices = df.index[trend_changes].tolist()

        if len(change_indices) > 0:
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

    # Bollinger Bands
    if show_bollinger and _bollinger_bands:
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

    # Fair Value Gaps
    if show_fvg and find_fair_value_gaps:
        fvgs = find_fair_value_gaps(df)
        for fvg in fvgs:
            gap_high = fvg["gap_high"]
            gap_low = fvg["gap_low"]
            start_date = fvg["start_date"]
            end_date = fvg["end_date"]

            if fvg["type"] == "bullish":
                fill_color = "rgba(40, 167, 69, 0.2)" if theme == "Light" else "rgba(0, 255, 136, 0.25)"
                line_color = "rgba(40, 167, 69, 0.6)" if theme == "Light" else "rgba(0, 255, 136, 0.7)"
            else:
                fill_color = "rgba(220, 53, 69, 0.2)" if theme == "Light" else "rgba(255, 68, 68, 0.25)"
                line_color = "rgba(220, 53, 69, 0.6)" if theme == "Light" else "rgba(255, 68, 68, 0.7)"

            fig.add_shape(
                type="rect",
                x0=start_date, y0=gap_low,
                x1=end_date, y1=gap_high,
                fillcolor=fill_color,
                line=dict(color=line_color, width=1, dash="dot"),
                layer="below",
                row=row, col=1
            )

            mid_date = start_date + (end_date - start_date) / 2 if isinstance(end_date, pd.Timestamp) else start_date
            mid_price = (gap_high + gap_low) / 2
            fig.add_annotation(
                x=mid_date, y=mid_price,
                text=f"FVG ({fvg['type']})",
                showarrow=False,
                font=dict(size=9, color=line_color),
                bgcolor=fill_color,
                bordercolor=line_color,
                borderwidth=1,
                opacity=0.8,
                row=row, col=1
            )

    # Support and Resistance
    if show_support_resistance and find_support_resistance:
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

    # Trade visualization zones
    if trade_entry and trade_stop_loss and trade_target and trade_direction:
        start_date = df.index[0]
        end_date = df.index[-1]

        if trade_direction == "LONG":
            risk_fill = "rgba(243, 18, 96, 0.2)" if theme == "Dark" else "rgba(220, 53, 69, 0.15)"
            risk_line = "#f31260" if theme == "Dark" else "#dc3545"
            reward_fill = "rgba(0, 255, 136, 0.2)" if theme == "Dark" else "rgba(40, 167, 69, 0.15)"
            reward_line = "#00ff88" if theme == "Dark" else "#28a745"

            fig.add_shape(type="rect", x0=start_date, y0=trade_stop_loss,
                         x1=end_date, y1=trade_entry,
                         fillcolor=risk_fill,
                         line=dict(color=risk_line, width=1, dash="dot"),
                         layer="below", row=row, col=1)

            fig.add_shape(type="rect", x0=start_date, y0=trade_entry,
                         x1=end_date, y1=trade_target,
                         fillcolor=reward_fill,
                         line=dict(color=reward_line, width=1, dash="dot"),
                         layer="below", row=row, col=1)

        elif trade_direction == "SHORT":
            risk_fill = "rgba(243, 18, 96, 0.2)" if theme == "Dark" else "rgba(220, 53, 69, 0.15)"
            risk_line = "#f31260" if theme == "Dark" else "#dc3545"
            reward_fill = "rgba(0, 255, 136, 0.2)" if theme == "Dark" else "rgba(40, 167, 69, 0.15)"
            reward_line = "#00ff88" if theme == "Dark" else "#28a745"

            fig.add_shape(type="rect", x0=start_date, y0=trade_entry,
                         x1=end_date, y1=trade_stop_loss,
                         fillcolor=risk_fill,
                         line=dict(color=risk_line, width=1, dash="dot"),
                         layer="below", row=row, col=1)

            fig.add_shape(type="rect", x0=start_date, y0=trade_target,
                         x1=end_date, y1=trade_entry,
                         fillcolor=reward_fill,
                         line=dict(color=reward_line, width=1, dash="dot"),
                         layer="below", row=row, col=1)

        # Add horizontal lines
        entry_color = "#00ff88" if theme == "Dark" else "#28a745" if trade_direction == "LONG" else "#f31260" if trade_direction == "SHORT" else "#b0b0b0"
        fig.add_hline(y=trade_entry, line_dash="solid", line_color=entry_color,
                     line_width=2, opacity=0.8,
                     annotation_text=f"Entry: ${trade_entry:.2f}",
                     annotation_position="right", row=row, col=1)
        fig.add_hline(y=trade_stop_loss, line_dash="dot",
                     line_color="#ff4444" if theme == "Dark" else "#dc3545",
                     line_width=1.5, opacity=0.7,
                     annotation_text=f"Stop Loss: ${trade_stop_loss:.2f}",
                     annotation_position="right", row=row, col=1)
        fig.add_hline(y=trade_target, line_dash="dot",
                     line_color="#00ff88" if theme == "Dark" else "#28a745",
                     line_width=1.5, opacity=0.7,
                     annotation_text=f"Target: ${trade_target:.2f}",
                     annotation_position="right", row=row, col=1)

    # Add cross markers
    if "SMA20" in df.columns and "SMA200" in df.columns:
        add_cross_markers(fig, df, price_col="close", s20="SMA20", s200="SMA200", row=row, col=1)

    # Pretouch band
    if pretouch_pct and pretouch_pct > 0 and "SMA200" in df.columns:
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
        colors = ["#28a745" if df["close"].iloc[i] >= df["open"].iloc[i] else "#dc3545"
                 for i in range(len(df))] if theme == "Light" else \
                ["#17c964" if df["close"].iloc[i] >= df["open"].iloc[i] else "#f31260"
                 for i in range(len(df))]
        vol_sma_color = "#495057" if theme == "Light" else "#888"

        fig.add_trace(go.Bar(
            x=df.index, y=df["volume"], name="Volume",
            marker_color=colors, opacity=0.6
        ), row=row, col=1)

        if _volume_sma:
            vol_sma = _volume_sma(df["volume"], window=20)
            fig.add_trace(go.Scatter(
                x=df.index, y=vol_sma, mode="lines", name="Vol SMA 20",
                line=dict(width=1, color=vol_sma_color)
            ), row=row, col=1)

        fig.update_yaxes(title_text="Volume", row=row, col=1)

    # RSI subplot
    if show_rsi and _rsi:
        row += 1
        rsi = _rsi(df["close"], window=14)
        rsi_color = "#6f42c1" if theme == "Light" else "#9b59b6"
        fig.add_trace(go.Scatter(
            x=df.index, y=rsi, mode="lines", name="RSI",
            line=dict(width=2, color=rsi_color)
        ), row=row, col=1)

        overbought_color = "#dc3545" if theme == "Light" else "red"
        oversold_color = "#28a745" if theme == "Light" else "green"
        neutral_color = "#6c757d" if theme == "Light" else "gray"
        fig.add_hline(y=70, line_dash="dash", line_color=overbought_color, opacity=0.5, row=row, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color=oversold_color, opacity=0.5, row=row, col=1)
        fig.add_hline(y=50, line_dash="dot", line_color=neutral_color, opacity=0.3, row=row, col=1)

        fig.update_yaxes(title_text="RSI", range=[0, 100], row=row, col=1)

    # MACD subplot
    if show_macd and _macd:
        row += 1
        if macd_mode == "Normal":
            macd_line, signal_line, histogram = _macd(df["close"])
            macd_title = "MACD"
        elif _macd_extended:
            macd_line, signal_line, histogram = _macd_extended(
                df["close"],
                sideways_window=macd_sideways_window,
                sideways_threshold=macd_sideways_threshold / 100.0
            )
            macd_title = "Extended MACD"
        else:
            macd_line, signal_line, histogram = _macd(df["close"])
            macd_title = "MACD"

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

        colors_hist = ["#28a745" if h >= 0 else "#dc3545" for h in histogram] if theme == "Light" else \
                     ["#17c964" if h >= 0 else "#f31260" for h in histogram]
        fig.add_trace(go.Bar(
            x=df.index, y=histogram, name="Histogram",
            marker_color=colors_hist, opacity=0.6
        ), row=row, col=1)

        zero_line_color = "#6c757d" if theme == "Light" else "gray"
        fig.add_hline(y=0, line_dash="dot", line_color=zero_line_color, opacity=0.3, row=row, col=1)
        fig.update_yaxes(title_text="MACD", row=row, col=1)

    # Update layout
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
            font_size=20,
            font_family="Arial, sans-serif",
            font_color=hover_text
        ),
        font=dict(color=text_color, size=12)
    )

    # Update grid colors
    for i in range(1, num_subplots + 1):
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor=grid_color, row=i, col=1)
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor=grid_color, row=i, col=1)

    return fig
