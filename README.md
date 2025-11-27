# TraderQ - Professional Trading Analytics Platform

A cloud-native, multi-user Streamlit application for tracking SMA (Simple Moving Average) crossovers, technical indicators, and portfolio management for stocks and cryptocurrencies.

**🔥 Now with Firebase Integration!**
- Multi-user support with authentication
- Cloud data storage (Firestore)
- Real-time data synchronization
- Free deployment on Streamlit Community Cloud

## Features

- **SMA Tracking**: Tracks 20-day and 200-day moving averages
- **Cross Detection**: Automatically detects Golden Cross (bullish) and Death Cross (bearish) signals
- **Multi-Asset Support**: Works with stocks, indices, and cryptocurrencies
- **Interactive Charts**: Beautiful Plotly charts with cross markers
- **Volume Analysis**: Volume bars with volume moving averages and volume ratios
- **RSI Indicator**: Relative Strength Index with overbought (70) and oversold (30) levels
- **MACD Indicator**: Moving Average Convergence Divergence with signal line and histogram
- **Extended MACD Indicator**: MACD that flattens during sideways/range-bound markets (adjustable lookback and threshold)
- **Bollinger Bands**: Price volatility bands with position tracking
- **EMA Indicator**: Exponential Moving Average with configurable period
- **Supertrend Indicator**: Trend-following indicator with color-coded bullish/bearish signals
- **Fair Value Gap (FVG) Indicator**: Identifies unfilled price gaps that often act as support/resistance
- **AI Recommendations**: Customizable indicator selection for AI analysis (SMA, RSI, MACD, Supertrend, FVG, News)
- **Trade Visualization**: Visual trade zones on charts showing risk (red) and reward (green) areas
- **Custom Tickers**: Add and persist your own ticker symbols
- **Trend Indicators**: Visual trend chips showing Bullish/Bearish status
- **Toggle Controls**: Enable/disable indicators in the sidebar

## 🔥 Firebase Features

- **User Authentication**: Secure email/password authentication
- **Cloud Data Storage**: All data stored in Google Firestore
- **Multi-User Support**: Each user has their own isolated data
- **Real-Time Sync**: Data updates instantly across devices
- **Data Persistence**: Never lose your tickers, alerts, or portfolio
- **Secure**: Firestore security rules ensure data privacy
- **Free Tier**: Firebase Spark Plan provides generous free limits

## 🚀 Quick Start

**New to TraderQ? Start here!**

1. **Prerequisites**: Python 3.8+, Node.js, Google account
2. **Follow the Quick Start Guide**: See [QUICKSTART.md](./QUICKSTART.md)
3. **Setup takes 15-20 minutes**

### Quick Install

```bash
# Install dependencies
pip install -r requirements.txt

# Verify setup
python verify_setup.py

# Follow QUICKSTART.md for Firebase setup
```

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

### Python Packages
- Python 3.8+
- Streamlit 1.39.0
- yfinance 0.2.44
- pandas 2.2.2
- numpy 1.26.4
- plotly 5.24.1
- firebase-admin 6.0.0+
- google-cloud-firestore 2.0.0+
- See [requirements.txt](./requirements.txt) for full list

### External Services
- **Firebase Project** (free tier available)
- **Firebase CLI** (install via npm)
- **Google account** (for Firebase Console)

## Data Sources

- **Stocks/Indices**: Yahoo Finance (via yfinance) or Google Finance (web scraping)
- **Cryptocurrencies**: CoinGecko API with Yahoo Finance fallback
- **User Data**: Google Cloud Firestore

## 📚 Documentation

- **[QUICKSTART.md](./QUICKSTART.md)** - 15-minute setup guide (START HERE!)
- **[FIREBASE_SETUP.md](./FIREBASE_SETUP.md)** - Detailed Firebase configuration
- **[DEPLOYMENT.md](./DEPLOYMENT.md)** - Streamlit Cloud deployment guide
- **[SETUP_CHECKLIST.md](./SETUP_CHECKLIST.md)** - Step-by-step checklist
- **[ALERTS_GUIDE.md](./ALERTS_GUIDE.md)** - Price alerts documentation
- **[FEATURE_LOCATIONS.md](./FEATURE_LOCATIONS.md)** - Feature reference

## 🛠️ Helper Scripts

- **`verify_setup.py`** - Check if setup is complete
- **`data_migration.py`** - Migrate JSON data to Firestore
- **`convert_key_to_toml.py`** - Convert service account key for Streamlit Cloud

## 💰 Cost

**Free Tier (Recommended for personal use):**
- Firebase Spark Plan: $0 (50K reads/day, 20K writes/day, 1GB storage)
- Streamlit Community Cloud: $0 (unlimited public apps)
- **Total: $0/month**

**Paid Tier (If you exceed free limits):**
- Firebase Blaze Plan: Pay as you go (~$5-15/month for moderate use)
- Streamlit Pro: $20/month (for private apps)

## 🔒 Security

- Firestore security rules ensure users can only access their own data
- Service account keys are never committed to Git (.gitignore)
- Passwords are hashed by Firebase Authentication
- All data encrypted in transit and at rest

## Version

Current version: v2.5.1

**Recent updates:**
- v2.5.1: Firebase integration with multi-user support
- v2.5.0: Extended MACD indicator with sideways detection
- v2.4.0: Fair Value Gap indicator and trade visualization

## License

This project is open source and available for personal use.

## 🆘 Support

1. Run `python verify_setup.py` to diagnose issues
2. Check [QUICKSTART.md](./QUICKSTART.md) for common solutions
3. Review Firebase Console for errors
4. Check Streamlit Cloud logs if deployed

## 🙏 Acknowledgments

- Built with [Streamlit](https://streamlit.io/)
- Data from [Yahoo Finance](https://finance.yahoo.com/) and [CoinGecko](https://www.coingecko.com/)
- Backend powered by [Google Firebase](https://firebase.google.com/)

