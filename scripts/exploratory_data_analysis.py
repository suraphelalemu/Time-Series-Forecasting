import warnings
import logging
import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
import seaborn as sns
import numpy as np
import math

from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose


class StockDataExploration:
    def __init__(
        self,
        data_dir: str = "../data",
        log_file: str = "../logs/exploratory_data_analysis.log",
        log_level: int = logging.INFO,
    ):
        """
        Initializes the StockDataExploration instance.

        Parameters:
        - data_dir (str): Directory to save downloaded data. Defaults to "../data".
        - log_file (str): Log file path. Defaults to "../logs/exploratory_data_analysis.log".
        - log_level (int): Logging level. Defaults to logging.INFO.
        """
        self.data_dir = data_dir
        os.makedirs(self.data_dir, exist_ok=True)
        self.logger = self._setup_logging(log_file, log_level)
        self.logger.info(
            "🚀 StockDataExploration initialized! Ready to analyze stock data."
        )

    def _setup_logging(self, log_file, log_level):
        """Sets up logging to prevent duplicate handlers."""
        os.makedirs(os.path.dirname(log_file), exist_ok=True)

        logger = logging.getLogger(__name__)
        if not logger.hasHandlers():
            logger.setLevel(log_level)

            file_handler = logging.FileHandler(log_file, encoding="utf-8")
            file_handler.setLevel(log_level)
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

            console_handler = logging.StreamHandler()
            console_handler.setLevel(log_level)
            console_formatter = logging.Formatter("%(message)s")
            console_handler.setFormatter(console_formatter)
            logger.addHandler(console_handler)

        return logger

    def plot_closing_price(self, data_dict):
        """Plots the closing price over time to identify trends and patterns."""
        rcParams["font.family"] = "sans-serif"
        rcParams["font.sans-serif"] = ["Segoe UI Emoji", "DejaVu Sans"]
        for symbol, df in data_dict.items():
            if df is None or df.empty:
                self._log_error(f"📉 DataFrame for {symbol} is empty.")
                return
            try:
                plt.figure(figsize=(10, 6))
                plt.plot(
                    df.index, df["Close"], label=f"{symbol} Closing Price", color="blue"
                )
                plt.title(f"📈 {symbol} Closing Price Over Time")
                plt.xlabel("Date")
                plt.ylabel("Price")
                plt.legend()
                plt.grid(True)
                plt.show()
                self.logger.info(f"✅ Successfully plotted closing price for {symbol}.")

            except Exception as e:
                self._log_error(
                    f"⚠️ Error plotting closing price for {symbol}: {str(e)}"
                )

    def plot_percentage_change(self, data_dict):
        """Plots the daily percentage change in the closing price to observe volatility."""
        rcParams["font.family"] = "sans-serif"
        rcParams["font.sans-serif"] = ["Segoe UI Emoji", "DejaVu Sans"]
        for symbol, df in data_dict.items():
            if df is None or df.empty:
                self._log_error(f"📉 DataFrame for {symbol} is empty.")
                return
            try:
                df["Pct_Change"] = df["Close"].pct_change() * 100
                plt.figure(figsize=(10, 6))
                plt.plot(
                    df.index,
                    df["Pct_Change"],
                    label=f"{symbol} Daily Percentage Change",
                    color="#1f77b4",
                )
                plt.title(f"📉 {symbol} Daily Percentage Change Over Time")
                plt.xlabel("Date")
                plt.ylabel("Percentage Change (%)")
                plt.legend()
                plt.grid(True)
                plt.show()
                self.logger.info(
                    f"✅ Successfully plotted percentage change for {symbol}."
                )

            except Exception as e:
                self._log_error(
                    f"⚠️ Error plotting percentage change for {symbol}: {str(e)}"
                )

    def analyze_price_trend(self, data_dict, window_size=30):
        """Plots the closing price, rolling mean, and volatility (rolling std) over time for multiple symbols."""
        sns.set(style="whitegrid")
        rcParams["font.family"] = "sans-serif"
        rcParams["font.sans-serif"] = ["Segoe UI Emoji", "DejaVu Sans"]
        for symbol, df in data_dict.items():
            try:
                if df is None or df.empty:
                    self._log_error(f"📉 DataFrame for {symbol} is empty.")
                    continue

                if "Close" not in df.columns:
                    self._log_error(
                        f"❌ 'Close' column not found in DataFrame for {symbol}."
                    )
                    continue

                # Calculate rolling mean and standard deviation (volatility)
                df["Rolling_Mean"] = df["Close"].rolling(window=window_size).mean()
                df["Rolling_Std"] = df["Close"].rolling(window=window_size).std()

                # Plotting each symbol in a separate figure for clarity
                plt.figure(figsize=(12, 6))

                # Closing price line
                sns.lineplot(
                    data=df,
                    x=df.index,
                    y="Close",
                    label=f"{symbol} Closing Price",
                    color="blue",
                    linestyle="solid",
                )
                # Rolling mean line
                sns.lineplot(
                    data=df,
                    x=df.index,
                    y="Rolling_Mean",
                    label=f"{symbol} {window_size}-day Rolling Mean",
                    color="orange",
                    linestyle="--",
                )
                # Rolling standard deviation (volatility) line
                sns.lineplot(
                    data=df,
                    x=df.index,
                    y="Rolling_Std",
                    label=f"{symbol} {window_size}-day Rolling Volatility",
                    color="green",
                    linestyle=":",
                )

                # Titles and labels
                plt.title(
                    f"📊 Closing Price Trend, Rolling Mean and Volatility of {symbol} Over Time",
                    fontsize=16,
                )
                plt.xlabel("Date", fontsize=12)
                plt.ylabel("Value", fontsize=12)
                plt.legend(title="Symbols")
                plt.grid(True)

                # Optional y-axis ticks increment adjustment for readability
                y_max = int(plt.ylim()[1])
                plt.yticks(range(0, y_max + 50, 50))  # Adjusts y-ticks by 50 increments

                # Tight layout for clarity
                plt.tight_layout()
                plt.show()

                self.logger.info(f"✅ Successfully plotted data for {symbol}.")

            except Exception as e:
                self._log_error(f"⚠️ Error plotting data for {symbol}: {str(e)}")

    def detect_outliers(self, data_dict):
        """Detects and highlights outliers in closing prices using the IQR method."""
        rcParams["font.family"] = "sans-serif"
        rcParams["font.sans-serif"] = ["Segoe UI Emoji", "DejaVu Sans"]
        for symbol, df in data_dict.items():
            try:
                Q1 = df["Close"].quantile(0.25)
                Q3 = df["Close"].quantile(0.75)
                IQR = Q3 - Q1
                outliers = df[
                    (df["Close"] < Q1 - 1.5 * IQR) | (df["Close"] > Q3 + 1.5 * IQR)
                ]
                plt.figure(figsize=(8, 6))
                sns.boxplot(y=df["Close"])
                plt.scatter(
                    outliers.index, outliers["Close"], color="red", label="Outliers"
                )
                plt.title(f"🚨 {symbol} Closing Price Outliers")
                plt.legend()
                plt.show()
                self.logger.info(f"✅ Successfully detected outliers for {symbol}.")

            except Exception as e:
                self._log_error(f"⚠️ Error detecting outliers for {symbol}: {str(e)}")

    def plot_unusual_daily_return(self, data_dict, threshold=2.5):
        """Analyzes days with unusually high or low returns."""
        rcParams["font.family"] = "sans-serif"
        rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans"]  # Changed font
        sns.set(style="whitegrid")
        warnings.filterwarnings(
            "ignore", category=UserWarning, message="Glyph 128201.*missing from font.*"
        )

        for symbol, df in data_dict.items():
            try:
                if df is None or df.empty:
                    self._log_error(f"📉 DataFrame for {symbol} is empty.")
                    continue

                df["Daily_Return"] = df["Close"].pct_change() * 100
                mean_return = df["Daily_Return"].mean()
                std_dev = df["Daily_Return"].std()
                unusual_returns = df[
                    (df["Daily_Return"] > mean_return + threshold * std_dev)
                    | (df["Daily_Return"] < mean_return - threshold * std_dev)
                ]

                plt.figure(figsize=(12, 6))
                sns.lineplot(
                    x=df.index,
                    y=df["Daily_Return"],
                    label=f"{symbol} Daily Return",
                    color="#1f77b4",
                )

                if not unusual_returns.empty:  # Check for unusual returns
                    plt.scatter(
                        unusual_returns.index,
                        unusual_returns["Daily_Return"],
                        color="red",
                        label=f"Unusual Returns (±{threshold}σ)",
                        s=50,
                        marker="o",
                    )

                plt.title(f"📉 Daily Returns with Unusual Days Highlighted - {symbol}")
                plt.xlabel("Date")
                plt.ylabel("Daily Return (%)")
                plt.axhline(0, color="grey", linestyle="--")
                plt.legend()
                plt.grid(True)
                plt.show()
                self.logger.info(
                    f"✅ Successfully plotted unusual daily returns for {symbol}."
                )

            except Exception as e:
                self._log_error(
                    f"⚠️ Error plotting unusual daily returns for {symbol}: {str(e)}"
                )

    def analyze_seasonality(self, data_dict, threshold=0.05):
        """Analyzes seasonality and trends using decomposition and stationarity tests."""
        sns.set(style="whitegrid")
        for symbol, df in data_dict.items():
            try:
                if df is None or df.empty:
                    self._log_error(f"DataFrame for {symbol} is empty.")
                    continue
                p_value = adfuller(df["Close"].dropna())[1]
                if p_value > threshold:
                    df["Close"] = df["Close"].diff().dropna()
                decomposition = seasonal_decompose(
                    df["Close"].dropna(), model="additive", period=252
                )
                plt.figure(figsize=(12, 8))
                plt.subplot(411)
                plt.plot(df["Close"], label=f"{symbol} Closing Price")
                plt.subplot(412)
                plt.plot(decomposition.trend, label=f"{symbol} Trend", color="orange")
                plt.subplot(413)
                plt.plot(
                    decomposition.seasonal, label=f"{symbol} Seasonal", color="green"
                )
                plt.subplot(414)
                plt.plot(decomposition.resid, label=f"{symbol} Residual", color="red")
                plt.tight_layout()
                plt.show()
            except Exception as e:
                self._log_error(f"Error analyzing seasonality for {symbol}: {str(e)}")

    def adf_test(self, series):
        """Perform ADF test for stationarity."""
        adf_result = adfuller(series.dropna())
        return adf_result[1]  
    
    def difference_series(self, series):
        """Apply differencing to the series to make it stationary."""
        return series.diff().dropna()
    
    def decompose_series(self, series, model='addictive'):
        """Decompose the time series into trend, seasonal, and residual components."""
        decomposition = seasonal_decompose(series.dropna(), model=model, period=252) 
        return decomposition    
    
    # def analyze_trends_and_seasonality(self, data_dict, threshold=0.05):
    #     """Analyze seasonality and trends of Tesla stock price by decomposing it."""
    #     sns.set(style="whitegrid")

    #     for symbol, df in data_dict.items():
    #         try:
    #             if df is None or df.empty:
    #                 self._log_error(f"DataFrame for {symbol} is empty.")
    #                 continue

    #             # Perform ADF test for stationarity
    #             p_value = self.adf_test(df['Close'])
                
    #             print(f"ADF test p-value for {symbol}: {p_value}")

    #             # If the p-value is greater than the threshold, apply differencing
    #             if p_value > threshold:
    #                 print(f"{symbol} series is non-stationary. Differencing the series.")
    #                 df['Close'] = self.difference_series(df['Close'])

    #             # After differencing, check again
    #             p_value = self.adf_test(df['Close'])
    #             print(f"ADF test p-value after differencing for {symbol}: {p_value}")

    #             # Decompose the series into trend, seasonal, and residual components
    #             decomposition = self.decompose_series(df['Close'])

    #             # Plot the decomposition results
    #             plt.figure(figsize=(12, 8))
    #             plt.subplot(411)
    #             plt.plot(df['Close'], label=f'{symbol} Closing Price')
    #             plt.title(f'{symbol} Closing Price')
    #             plt.legend(loc='best')

    #             plt.subplot(412)
    #             plt.plot(decomposition.trend, label=f'{symbol} Trend', color='orange')
    #             plt.title(f'{symbol} Trend')
    #             plt.legend(loc='best')

    #             plt.subplot(413)
    #             plt.plot(decomposition.seasonal, label=f'{symbol} Seasonal', color='green')
    #             plt.title(f'{symbol} Seasonal')
    #             plt.legend(loc='best')

    #             plt.subplot(414)
    #             plt.plot(decomposition.resid, label=f'{symbol} Residual', color='red')
    #             plt.title(f'{symbol} Residual')
    #             plt.legend(loc='best')

    #             plt.tight_layout()
    #             plt.show()

    #         except Exception as e:
    #             self._log_error(f"Error analyzing {symbol}: {str(e)}")

    def analyze_trends_and_seasonality(self, data_dict, threshold=0.05):
        """Analyze seasonality and trends of stock prices by decomposing them."""
        sns.set(style="whitegrid")

        # Define a pastel color palette for the plots
        colors = {
            'closing_price': '#1f77b4',  
            'trend': '#ff7f0e',           
            'seasonal': '#2ca02c',        
            'residual': '#FF677D'        
        }

        for symbol, df in data_dict.items():
            try:
                if df is None or df.empty:
                    self._log_error(f"DataFrame for {symbol} is empty.")
                    continue

                # Perform ADF test for stationarity
                p_value = self.adf_test(df['Close'])
                
                print(f"ADF test p-value for {symbol}: {p_value}")

                # If the p-value is greater than the threshold, apply differencing
                if p_value > threshold:
                    print(f"{symbol} series is non-stationary. Differencing the series.")
                    df['Close'] = self.difference_series(df['Close'])

                # After differencing, check again
                p_value = self.adf_test(df['Close'])
                print(f"ADF test p-value after differencing for {symbol}: {p_value}")

                # Decompose the series into trend, seasonal, and residual components
                decomposition = self.decompose_series(df['Close'])

                # Plot the decomposition results
                plt.figure(figsize=(14, 10))

                plt.subplot(411)
                plt.plot(df['Close'], label=f'{symbol} Closing Price', color=colors['closing_price'])
                plt.title(f'{symbol} Closing Price', fontsize=16)
                plt.legend(loc='best')
                plt.grid(axis='y', linestyle='--')

                plt.subplot(412)
                plt.plot(decomposition.trend, label=f'{symbol} Trend', color=colors['trend'])
                plt.title(f'{symbol} Trend', fontsize=16)
                plt.legend(loc='best')
                plt.grid(axis='y', linestyle='--')

                plt.subplot(413)
                plt.plot(decomposition.seasonal, label=f'{symbol} Seasonal', color=colors['seasonal'])
                plt.title(f'{symbol} Seasonal', fontsize=16)
                plt.legend(loc='best')
                plt.grid(axis='y', linestyle='--')

                plt.subplot(414)
                plt.plot(decomposition.resid, label=f'{symbol} Residual', color=colors['residual'])
                plt.title(f'{symbol} Residual', fontsize=16)
                plt.legend(loc='best')
                plt.grid(axis='y', linestyle='--')

                plt.tight_layout()
                plt.show()

            except Exception as e:
                self._log_error(f"Error analyzing {symbol}: {str(e)}")


    def _log_error(self, message):
        """Logs error messages to the log file."""
        if self.logger:
            self.logger.error(message)
        print(f"🚨 Error: {message}")