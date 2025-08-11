## Time Series Forecasting for Portfolio Management Optimization
### Overview
This project implements a comprehensive portfolio management system using time series forecasting techniques to optimize asset allocation. The system analyzes historical financial data for three key assets (TSLA, BND, SPY) to predict market trends, assess risk, and recommend optimal portfolio strategies.

### Key Features
Data Acquisition & Preprocessing: Automated retrieval and cleaning of historical financial data

Exploratory Data Analysis: Comprehensive visualization of price trends, returns, and volatility

Time Series Forecasting: Implementation of ARIMA and LSTM models for price prediction

Risk Assessment: Calculation of key risk metrics including Value-at-Risk and Sharpe Ratio

Portfolio Optimization: Modern Portfolio Theory implementation to identify optimal asset allocations

Performance Backtesting: Historical simulation of portfolio performance

### Implementation Details
1. Data Pipeline
python
# Fetch and preprocess data
tickers = ['TSLA', 'BND', 'SPY']
data = yf.download(tickers, start='2015-07-01', end='2025-07-31')

# Handle MultiIndex structure
closing_prices = data['Close'].copy() if isinstance(data.columns, pd.MultiIndex) else data[[f'{t}.Close' for t in tickers]].copy()
Automatically adapts to different yfinance data formats

Robust missing value handling with forward-filling

Normalization and feature engineering for modeling

2. Time Series Analysis
Stationarity Testing
python
def test_stationarity(timeseries):
    dftest = adfuller(timeseries, autolag='AIC')
    # Returns test statistics and critical values
Volatility Analysis
Rolling 21-day volatility (annualized)

#Return distribution analysis

Outlier detection using z-scores

3. Forecasting Models
ARIMA/SARIMA
python
model = auto_arima(returns, seasonal=False, trace=True)
forecast = model.predict(n_periods=5)
LSTM Neural Network
python
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(seq_length, 1)))
model.add(Dense(1))
4. Portfolio Optimization
python
# Calculate efficient frontier
mu = expected_returns.mean_historical_return(prices)
S = risk_models.sample_cov(prices)
ef = EfficientFrontier(mu, S)
Implements Modern Portfolio Theory

Maximizes Sharpe ratio

Provides optimal asset weights

How to Use
Install dependencies:

bash
pip install -r requirements.txt
Run the analysis:

bash
python portfolio_analysis.py
View results:

Generated visualizations in /plots

Model performance metrics in console output

Portfolio recommendations in results.csv

# Key Findings
Asset Characteristics:

TSLA shows high volatility (σ ≈ 50%) with significant return outliers

BND provides stable returns with low volatility (σ ≈ 5%)

SPY offers balanced risk-return profile (σ ≈ 15%)

# Stationarity:

Price series are non-stationary (ADF p > 0.05)

Return series are stationary (ADF p < 0.01)

# Optimal Portfolio:

Recommended allocation varies with risk tolerance

Maximum Sharpe portfolio typically includes all three assets

Future Enhancements
Incorporate alternative data sources (news sentiment, macroeconomic indicators)

Implement more sophisticated deep learning architectures

Add transaction cost modeling

Develop interactive dashboard for visualization

# Dependencies
Python 3.8+

yfinance, pandas, numpy

statsmodels, pmdarima

tensorflow, scikit-learn

PyPortfolioOpt