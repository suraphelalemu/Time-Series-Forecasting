import logging
import os
import pandas as pd
import pynance as pn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import math


class DataPreprocessor:
    """
    DataPreprocessor class for fetching, detecting, cleaning, and analyzing financial data from YFinance.
    """

    def __init__(
        self,
        data_dir: str = "../data",
        log_file: str = "../logs/data_preprocessing.log",
        log_level: int = logging.INFO,
    ):
        """
        Initializes the DataPreprocessor instance.

        Parameters:
        - data_dir (str): Directory to save downloaded data. Defaults to "../data".
        - log_file (str): Log file path. Defaults to "../logs/data_preprocessing.log".
        - log_level (int): Logging level. Defaults to logging.INFO.
        """
        self.data_dir = data_dir
        os.makedirs(self.data_dir, exist_ok=True)
        self.logger = self._setup_logging(log_file, log_level)

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

    def get_data(self, start_date, end_date, symbols):
        """
        Fetches historical data for each symbol and saves it as a CSV.

        Returns:
        - dict: Dictionary with symbol names as keys and file paths of saved CSV files as values.
        """
        data_paths = {}

        for symbol in symbols:
            try:
                self.logger.info(
                    f"📊 Fetching data for {symbol} from {start_date} to {end_date}..."
                )
                data = pn.data.get(symbol, start=start_date, end=end_date)
                file_path = os.path.join(self.data_dir, f"{symbol}.csv")
                data.to_csv(file_path)

                normalized_path = file_path.replace("\\", "/")
                data_paths[symbol] = normalized_path
                self.logger.info(f"✅ Data for {symbol} saved to '{normalized_path}'.")

            except ValueError as ve:
                error_message = f"⚠️ Data format issue for {symbol}: {ve}"
                self.logger.error(error_message)

            except Exception as e:
                error_message = f"❌ Failed to fetch data for {symbol}: {e}"
                self.logger.error(error_message)

        return data_paths

    def load_data(self, symbol):
        """
        📥 Loads data from a CSV file for a specified symbol.

        Parameters:
        - symbol (str): Stock symbol to load data for (e.g., "TSLA").

        Returns:
        - pd.DataFrame: DataFrame with loaded data, or raises FileNotFoundError if missing.
        """
        file_path = os.path.join(self.data_dir, f"{symbol}.csv")
        if os.path.exists(file_path):
            normalized_path = file_path.replace("\\", "/")
            self.logger.info(f"📊 Loading data for {symbol} from '{normalized_path}'.")
            return pd.read_csv(file_path, parse_dates=["Date"], index_col="Date")
        else:
            error_message = (
                f"❌ Data file for symbol '{symbol}' not found. Run `get_data()` first."
            )
            self.logger.error(error_message)
            raise FileNotFoundError(error_message)

    def inspect_data(self, data):
        """
        🔍 Inspects the data by checking data types, missing values, and duplicates.

        Parameters:
        - data (pd.DataFrame): DataFrame containing stock data for inspection.

        Returns:
        - dict: A dictionary containing the following inspection results:
        - Data types of the columns.
        - Missing values count.
        - Duplicate rows count.
        """
        # Perform data inspection
        inspection_results = {
            "data_types": data.dtypes,
            "missing_values": data.isnull().sum(),
            "duplicate_rows": data.duplicated().sum(),
        }

        self.logger.info(f"📋 Data inspection results:\n{inspection_results}")
        return inspection_results

    def analyze_data(self, data):
        """
        📊 Analyzes data by calculating basic statistics and checking for anomalies.

        Parameters:
        - data (pd.DataFrame): DataFrame containing stock data for analysis.

        Returns:
        - dict: Summary statistics including:
        - 📈 Mean
        - 📉 Median
        - 📊 Standard Deviation
        - ❓ Count of Missing Values
        """
        # 🧮 Calculate basic statistics
        analysis_results = {
            "mean": data.mean(),
            "median": data.median(),
            "std_dev": data.std(),
            "missing_values": data.isnull().sum(),
        }

        # 📜 Log the analysis results for reference
        self.logger.info(
            f"✨ Basic statistics calculated for data:\n{analysis_results}"
        )

        return analysis_results

    def detect_outliers(self, data, method="iqr", z_threshold=3):
        """
        🔍 Detects outliers in the stock data using either the IQR or Z-score method.

        Parameters:
        - data (pd.DataFrame): DataFrame containing stock data.
        - method (str): Outlier detection method ('iqr' or 'z_score'). Default is 'iqr'.
        - z_threshold (int): Z-score threshold to classify an outlier. Default is 3 (only used if method is 'z_score').

        Returns:
        - pd.DataFrame: DataFrame containing boolean values indicating outliers.
        """
        # 🗂️ Initialize DataFrame for outliers
        outliers = pd.DataFrame(index=data.index)

        # 📊 Analyze each relevant column for outliers
        for col in ["Open", "High", "Low", "Close", "Adj Close", "Volume"]:
            if col in data.columns:
                if method == "z_score":
                    # 🧮 Calculate Z-scores
                    z_scores = np.abs((data[col] - data[col].mean()) / data[col].std())
                    outliers[col] = z_scores > z_threshold
                elif method == "iqr":
                    # 📏 Calculate IQR and detect outliers
                    Q1 = data[col].quantile(0.25)
                    Q3 = data[col].quantile(0.75)
                    IQR = Q3 - Q1
                    outliers[col] = (data[col] < (Q1 - 1.5 * IQR)) | (
                        data[col] > (Q3 + 1.5 * IQR)
                    )

        # 📋 Log the method used for outlier detection
        self.logger.info(f"✨ Outliers detected using {method} method.")

        return outliers

    def plot_outliers(self, data, outliers, symbol):
        """
        📈 Plots box plots to visualize outliers in the data.

        Parameters:
        - data (pd.DataFrame): DataFrame containing stock data.
        - outliers (pd.DataFrame): Boolean DataFrame indicating outliers.
        """
        rcParams["font.family"] = "sans-serif"
        rcParams["font.sans-serif"] = ["Segoe UI Emoji", "DejaVu Sans"]
        # 🔍 Identify columns with outliers
        columns_with_outliers = [
            col
            for col in data.columns
            if col in outliers.columns and outliers[col].any()
        ]

        if not columns_with_outliers:
            self.logger.info("✅ No outliers detected in any columns.")
            return

        num_plots = len(columns_with_outliers)
        grid_size = math.ceil(math.sqrt(num_plots))

        # 🎨 Set up the figure and axes for plotting with increased height
        fig, axes = plt.subplots(
            grid_size, grid_size, figsize=(12 * grid_size, 7 * grid_size)
        )
        fig.suptitle(
            f"📊 Outlier Detection for {symbol}", fontsize=18, fontweight="bold"
        )

        if num_plots == 1:
            axes = [axes]
        else:
            axes = axes.ravel()

        for i, col in enumerate(columns_with_outliers):
            ax = axes[i]
            ax.plot(data.index, data[col], label=col, color="#1f77b4", linewidth=2)
            ax.scatter(
                data.index[outliers[col]],
                data[col][outliers[col]],
                color="red",
                s=20,
                edgecolor="black",
                label="Outliers",
            )

            # 📋 Customize axes
            ax.set_title(f"{col} - Time Series with Outliers of {symbol}", fontsize=14)
            ax.set_xlabel("Date", fontsize=12)
            ax.set_ylabel(col, fontsize=12)
            ax.legend()
            ax.grid(True, linestyle="--", alpha=0.7)

        # 🔒 Hide any unused subplots
        for j in range(i + 1, len(axes)):
            axes[j].axis("off")

        plt.tight_layout()
        plt.show()

    def handle_outliers(self, data_dict, outliers_dict):
        """
        ✨ Handles detected outliers by replacing them with NaN for later filling.

        Parameters:
        - data_dict (dict): Dictionary containing stock data as DataFrames for each symbol.
        - outliers_dict (dict): Dictionary containing boolean DataFrames indicating positions of outliers.

        Returns:
        - dict: Dictionary with cleaned data for each symbol where outliers have been handled.
        """
        cleaned_data_dict = {}

        # 🔄 Iterate over each stock symbol and its corresponding data
        for symbol, data in data_dict.items():
            cleaned_data = data.copy()

            if symbol in outliers_dict:
                outliers = outliers_dict[symbol]
                cleaned_data[outliers] = np.nan

                # 🔄 Fill NaN values using interpolation, backfill, and forward fill
                cleaned_data.interpolate(method="time", inplace=True)
                cleaned_data.bfill(inplace=True)
                cleaned_data.ffill(inplace=True)

                self.logger.info(
                    f"🔧 Outliers handled for {symbol} by setting to NaN and filling with interpolation."
                )

            cleaned_data_dict[symbol] = cleaned_data

        self.logger.info("✅ Outliers handled across all data sources.")
        return cleaned_data_dict