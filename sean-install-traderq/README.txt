================================================================================
                    TraderQ - Installation & Usage Guide
                            Version 2.5.0
================================================================================

Thank you for using TraderQ! This guide will help you set up and run the app.

================================================================================
                            QUICK START
================================================================================

1. Run the Installer:
   - Double-click "INSTALL_TRADERQ.bat"
   - Follow the on-screen instructions
   - The installer will check if Python is installed and set everything up

2. Start TraderQ:
   - After installation, double-click "START_TRADERQ.bat"
   - A server window will open - KEEP IT OPEN while using TraderQ
   - Your browser will open automatically with TraderQ

3. To Stop TraderQ:
   - Press Ctrl+C in the server window
   - Close the server window

================================================================================
                            REQUIREMENTS
================================================================================

- Windows 10 or later
- Python 3.7 or later (the installer will check and guide you if missing)
- Internet connection (for stock/crypto data)

================================================================================
                       INSTALLATION STEPS
================================================================================

STEP 1: Install Python (if needed)
-----------------------------------
If Python is not installed:
1. The installer will open the Python download page automatically
2. Download Python 3.x (latest version)
3. IMPORTANT: During installation, check "Add Python to PATH"
4. Run INSTALL_TRADERQ.bat again after installing Python

STEP 2: Run the Installer
--------------------------
1. Double-click "INSTALL_TRADERQ.bat"
2. The installer will:
   - Check if Python is installed
   - Verify traderq.html exists
   - Create START_TRADERQ.bat launcher script

STEP 3: Launch TraderQ
-----------------------
1. Double-click "START_TRADERQ.bat"
2. A black command window will open - this is the server
3. KEEP THIS WINDOW OPEN while using TraderQ
4. Your browser will open automatically at http://localhost:8000/traderq.html

================================================================================
                          USING TRADERQ
================================================================================

After TraderQ opens in your browser:

1. Enter a Ticker Symbol:
   - Type a stock symbol (e.g., SPY, AAPL, MSFT)
   - Or a crypto symbol (e.g., BTC-USD, ETH-USD)
   - Click "Load Data"

2. View Charts:
   - The chart will display with technical indicators
   - Toggle indicators on/off in the sidebar
   - Switch between Dark/Light theme

3. Get AI Recommendations:
   - Click the "🤖 AI Recommendations" tab
   - Click "Generate AI Recommendation"
   - View trade parameters (entry, stop loss, target)
   - See trade zones visualized on the chart

4. Customize Indicators:
   - Use checkboxes in sidebar to show/hide indicators
   - Adjust Extended MACD settings if needed
   - Choose which indicators AI Recommendations should use

================================================================================
                          TROUBLESHOOTING
================================================================================

Problem: "Python is not recognized as a command"
Solution: 
- Install Python from https://www.python.org/downloads/
- Make sure to check "Add Python to PATH" during installation
- Restart your computer after installing Python
- Run INSTALL_TRADERQ.bat again

Problem: "This site cannot be reached" when opening TraderQ
Solution:
- Make sure START_TRADERQ.bat is running (you should see a server window)
- Keep the server window open - closing it stops the server
- Try http://localhost:8080/traderq.html if port 8000 doesn't work
- Check Windows Firewall isn't blocking Python

Problem: "Port 8000 is already in use"
Solution:
- Close other applications that might be using port 8000
- The launcher will automatically try port 8080 as a backup
- Or manually run: python -m http.server 8080

Problem: No data loading / CORS errors
Solution:
- Make sure you're accessing via http://localhost:8000 (not file://)
- Try switching to "Alpha Vantage" data source in the sidebar
- Get a free API key from alphavantage.co
- Or install a CORS browser extension

Problem: Charts not displaying correctly
Solution:
- Refresh the page (Ctrl+F5 or F5)
- Make sure you have internet connection
- Check browser console for errors (F12)

================================================================================
                          DATA SOURCES
================================================================================

Yahoo Finance (Default):
- Works when running from local server
- Free, no API key needed
- May have rate limits

Alpha Vantage (Alternative):
- Get free API key from: https://www.alphavantage.co/support/#api-key
- Switch data source in sidebar
- Enter API key in sidebar
- More reliable for production use

================================================================================
                            FEATURES
================================================================================

Technical Indicators:
- SMA 20/200 (Simple Moving Averages)
- EMA (Exponential Moving Average)
- RSI (Relative Strength Index)
- MACD (Normal and Extended modes)
- Supertrend
- Bollinger Bands
- Fair Value Gap (FVG)
- Volume Analysis

AI Recommendations:
- Customizable indicator selection
- Trade entry, stop loss, and target calculation
- Visual trade zones on charts (red = risk, green = reward)
- Risk/Reward ratio analysis
- Confidence scoring

Chart Features:
- Interactive Plotly charts
- Dark/Light themes
- Multiple timeframes (Daily, 5min, 15min, 1h, Weekly, Monthly)
- Golden/Death cross markers
- Support/Resistance detection

================================================================================
                            SUPPORT
================================================================================

If you encounter any issues:
1. Check the Troubleshooting section above
2. Make sure Python is installed and in PATH
3. Ensure traderq.html is in the same folder as the scripts
4. Try running START_TRADERQ.bat as Administrator

================================================================================
                          FILES IN THIS PACKAGE
================================================================================

- traderq.html           Main application file
- INSTALL_TRADERQ.bat    Installer script (run once)
- START_TRADERQ.bat      Launcher script (created by installer)
- README.txt            This file

================================================================================

Enjoy using TraderQ!

