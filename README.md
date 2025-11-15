# TraderQ - SMA 20/200 Tracker

A Streamlit-based web application for tracking SMA (Simple Moving Average) crossovers for stocks and cryptocurrencies.

## Features

- **SMA Tracking**: Tracks 20-day and 200-day moving averages
- **Cross Detection**: Automatically detects Golden Cross (bullish) and Death Cross (bearish) signals
- **Multi-Asset Support**: Works with stocks, indices, and cryptocurrencies
- **Interactive Charts**: Beautiful Plotly charts with cross markers
- **Volume Analysis**: Volume bars with volume moving averages and volume ratios
- **RSI Indicator**: Relative Strength Index with overbought (70) and oversold (30) levels
- **MACD Indicator**: Moving Average Convergence Divergence with signal line and histogram
- **Bollinger Bands**: Price volatility bands with position tracking
- **EMA Indicator**: Exponential Moving Average with configurable period
- **Supertrend Indicator**: Trend-following indicator with color-coded bullish/bearish signals
- **Custom Tickers**: Add and persist your own ticker symbols
- **Trend Indicators**: Visual trend chips showing Bullish/Bearish status
- **Toggle Controls**: Enable/disable indicators in the sidebar

## Installation

1. Clone the repository:
```bash
git clone https://github.com/dquillman/traderq-sma-tracker.git
cd traderq-sma-tracker
```

2. Create a virtual environment:
```bash
python -m venv .venv
```

3. Activate the virtual environment:
- Windows: `.venv\Scripts\activate`
- Linux/Mac: `source .venv/bin/activate`

4. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the Streamlit app:
```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

## Requirements

- Python 3.8+
- Streamlit 1.39.0
- yfinance 0.2.44
- pandas 2.2.2
- numpy 1.26.4
- plotly 5.24.1
- pandas-datareader
- pycoingecko

## Data Sources

- **Stocks/Indices**: Yahoo Finance (via yfinance)
- **Cryptocurrencies**: CoinGecko API with Yahoo Finance fallback

## Version

Current version: v2.3.0

## License

This project is open source and available for personal use.

