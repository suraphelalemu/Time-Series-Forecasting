import logging
import warnings
import pandas as pd
import os
import numpy as np
import pmdarima as pm
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
import joblib
from tensorflow.keras.models import save_model


class TimeSeriesModelTrainer:
    def __init__(
        self,
        data: pd.DataFrame,
        data_dir: str = "../data",
        log_file: str = "../logs/time_series_forecasting.log",
        log_level: int = logging.INFO,
        column: str = "Close",
    ):
        """
        Initialize the time series model trainer.

        Parameters:
        - data (pd.DataFrame): Time series data.
        - data_dir (str): Directory to save models or data. Defaults to "../data".
        - log_file (str): Log file path. Defaults to "../logs/time_series_forecasting.log".
        - log_level (int): Logging level. Defaults to logging.INFO.
        - column (str): The column name to forecast.
        """
        self.df = data
        self.column = column
        self.data_dir = data_dir
        os.makedirs(self.data_dir, exist_ok=True)
        self.logger = self._setup_logging(log_file, log_level)
        self.train = None
        self.test = None
        self.model = {}
        self.prediction = {}
        self.metric = {}
        self.scaler = MinMaxScaler(feature_range=(0, 1))  # Initialize the scaler

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

    def prepare_data(self, train_size=0.8):
        """
        🌟 Prepare the data for training and testing, and apply scaling. 📊
        """
        try:
            # 🕒 Ensure the index is in datetime format and resample the data daily
            self.df.index = pd.to_datetime(self.df.index)
            self.df = self.df.resample("D").ffill().dropna()

            # 🔍 Split the data into training and testing sets
            split_idx = int(len(self.df) * train_size)
            self.train, self.test = self.df[:split_idx].copy(), self.df[split_idx:].copy()
            self.logger.info(
                f"Data split: {len(self.train)} training samples, {len(self.test)} testing samples 🏋️‍♂️"
            )

            # 📏 Scale the specified column using .loc
            self.train.loc[:, self.column] = self.scaler.fit_transform(
                self.train[[self.column]]
            )
            self.test.loc[:, self.column] = self.scaler.transform(self.test[[self.column]])

            # ✅ Data preparation completed successfully!
            self.logger.info("Data preparation completed successfully! 🎉")
        except Exception as e:
            # 🚨 Log the error and raise an exception
            self.logger.error(f"Error in preparing data: {e} ❌")
            raise ValueError("Data preparation failed")        

    def train_arima(self):
        """
        📈 Train ARIMA model using auto_arima. 🚀
        """
        # 🔕 Suppress FutureWarnings to keep the output clean
        warnings.simplefilter(action='ignore', category=FutureWarning)

        try:
            self.logger.info("🔍 Starting to train the ARIMA model...")

            # 🏗️ Build the ARIMA model using auto_arima
            model = pm.auto_arima(
                self.train[self.column],
                seasonal=False,  # ❌ No seasonal component
                trace=True,      # 📊 Show progress
                error_action="ignore",
                suppress_warnings=True,  # 🤫 Suppress warnings during training
                stepwise=True,  # 🚶‍♂️ Stepwise search for optimal parameters
            )

            # 📦 Store the trained model
            self.model["ARIMA"] = model
            
            # 📜 Print the model summary
            print(model.summary())
            
            # 📢 Log the trained model parameters
            self.logger.info(
                f"✅ ARIMA model trained with parameters: {model.get_params()}"
            )
            
        except Exception as e:
            # 🚨 Log the error details and raise an exception
            self.logger.error(f"⚠️ Error in ARIMA training: {e} ❌")
            raise ValueError("ARIMA model training failed")

    def train_sarima(self, seasonal_period=5):
        """
        🌾 Train SARIMA model using auto_arima. 📈
        """
        try:
            self.logger.info("🔍 Training SARIMA model...")
            
            # 📊 Build the SARIMA model with specified parameters
            model = pm.auto_arima(
                self.train[self.column],
                seasonal=True,            # 🌱 Seasonal component included
                m=seasonal_period,       # 📅 Seasonal period
                start_p=0,               # Starting value for p
                start_q=0,               # Starting value for q
                max_p=3,                 # Maximum value for p
                max_q=3,                 # Maximum value for q
                d=1,                      # Non-seasonal differencing
                D=1,                      # Seasonal differencing
                trace=True,              # Show progress
                error_action="ignore",   # Ignore errors
                suppress_warnings=True,  # Suppress warnings
            )
            
            # 📦 Store the trained model
            self.model["SARIMA"] = model
            
            # 📜 Print the model summary
            print(model.summary())
            
            # 📢 Log the trained model parameters
            self.logger.info(
                f"✅ SARIMA model trained with parameters: {model.get_params()}"
            )
        except Exception as e:
            # 🚨 Log the error details and raise an exception
            self.logger.error(f"⚠️ Error in SARIMA training: {e} ❌")
            raise ValueError("SARIMA model training failed")
        

    def _create_sequence(self, data, seq_length=60):
        """
        🔄 Create sequences of data for LSTM. 📊
        
        Args:
            data (array-like): Input data for creating sequences.
            seq_length (int): Length of each sequence.
            
        Returns:
            tuple: Arrays of input sequences (xs) and corresponding targets (ys).
        """
        xs, ys = [], []  # 🗂️ Initialize lists to hold sequences and targets
        
        # 🔍 Create sequences of the specified length
        for i in range(len(data) - seq_length):
            x = data[i:i + seq_length]  # 📏 Extract the sequence
            y = data[i + seq_length]     # 🎯 Extract the target value
            xs.append(x)                 # 📥 Append sequence to xs
            ys.append(y)                 # 📥 Append target to ys
            
        return np.array(xs), np.array(ys)  # 🔙 Return as numpy arrays

    def train_lstm(self, seq_length=60, epochs=50, batch_size=32):
        """
        🧠 Train an LSTM model on the data. 📊
        """
        try:
            self.logger.info("🔍 Training LSTM model...")
            
            # 📊 Prepare the training data
            data = self.train[self.column].values.reshape(-1, 1)

            # 🔄 Create sequences for LSTM
            X_train, y_train = self._create_sequence(data, seq_length)

            # 🏗️ Build the LSTM model
            model = Sequential()
            model.add(Input(shape=(seq_length, 1)))
            model.add(LSTM(50, activation="relu", return_sequences=True))
            model.add(Dropout(0.2))  # 🌀 Dropout for regularization
            model.add(LSTM(50, activation="relu"))
            model.add(Dropout(0.2))
            model.add(Dense(1))  # 🎯 Output layer

            # 🛠️ Compile the model
            model.compile(optimizer="adam", loss="mse")
            model.summary()  # 📜 Print model summary

            # ⏳ Early stopping to prevent overfitting
            early_stopping = EarlyStopping(
                monitor="val_loss", patience=5, restore_best_weights=True
            )

            # 🚀 Fit the model
            history = model.fit(
                X_train,
                y_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_split=0.1,
                callbacks=[early_stopping],
                verbose=1,
            )

            # 📦 Store the trained model and history
            self.model["LSTM"] = {
                "model": model,
                "history": history,
                "seq_length": seq_length,
            }
            self.logger.info("✅ LSTM model training completed")

            # 📈 Plot training and validation loss
            self.plot_training_history(history)

        except Exception as e:
            # 🚨 Log the error details and raise an exception
            self.logger.error(f"⚠️ Error in LSTM training: {e} ❌")
            raise ValueError("LSTM model training failed")

    def plot_training_history(self, history):
        """
        📊 Plot the training and validation loss. 📈
        """
        try:
            plt.figure(figsize=(10, 6))
            plt.plot(history.history["loss"], label="Train Loss")
            plt.plot(history.history["val_loss"], label="Validation Loss")
            plt.title("Model Training and Validation Loss")
            plt.xlabel("Epochs")
            plt.ylabel("Loss")
            plt.legend()
            plt.show()  # 🖼️ Display the plot
            self.logger.info("✅ Training history plotted successfully")
        except Exception as e:
            # 🚨 Log the error details and raise an exception
            self.logger.error(f"⚠️ Error in plotting training history: {e} ❌")
            raise ValueError("Plotting training history failed")

    def make_prediction(self):
        """
        🔮 Generate predictions using all trained models. 📊
        """
        try:
            for model_name, model_data in self.model.items():
                if model_name == "ARIMA" or model_name == "SARIMA":
                    # 📈 Predict using ARIMA or SARIMA
                    self.prediction[model_name] = model_data.predict(
                        n_periods=len(self.test)
                    )
                elif model_name == "LSTM":
                    # 🌊 Prepare data for LSTM predictions
                    model = model_data["model"]
                    seq_length = model_data["seq_length"]
                    data = np.array(
                        self.train[self.column].values[-seq_length:].reshape(-1, 1)
                    )
                    predictions = []
                    
                    # 🔄 Predict for each test data point
                    for i in range(len(self.test)):
                        pred = model.predict(data.reshape(1, seq_length, 1))
                        predictions.append(pred[0, 0])
                        data = np.append(data[1:], pred[0, 0]).reshape(-1, 1)
                        
                    self.prediction["LSTM"] = np.array(predictions)

            self.logger.info("✅ Predictions generated for all models")
        except Exception as e:
            # 🚨 Log the error details and raise an exception
            self.logger.error(f"⚠️ Error in making predictions: {e} ❌")
            raise ValueError("Prediction generation failed")

    def evaluate_model(self):
        """
        📊 Evaluate all models and log metrics. 📈
        """
        try:
            metric_data = []
            for model_name, model in self.model.items():
                predictions = self.prediction.get(model_name)
                if predictions is None:
                    self.logger.error(f"No predictions for model {model_name}")
                    continue

                # 📏 Flatten the test data
                test_data = self.test[self.column].values
                mae = mean_absolute_error(test_data, predictions)
                rmse = np.sqrt(mean_squared_error(test_data, predictions))
                mape = np.mean(np.abs((test_data - predictions) / test_data)) * 100

                self.metric[model_name] = {"MAE": mae, "RMSE": rmse, "MAPE": mape}
                self.logger.info(
                    f"{model_name} - MAE: {mae:.2f}, RMSE: {rmse:.2f}, MAPE: {mape:.2f}%"
                )

                metric_data.append([model_name, mae, rmse, mape])

            # 📊 Display metrics in DataFrame
            metric_df = pd.DataFrame(
                metric_data, columns=["Model", "MAE", "RMSE", "MAPE"]
            )
            print("\n📈 Model Evaluation Metrics:\n", metric_df)
        except Exception as e:
            # 🚨 Log the error details and raise an exception
            self.logger.error(f"⚠️ Error in model evaluation: {e} ❌")
            raise ValueError("Model evaluation failed")

    def plot_result(self):
        """
        📊 Plot the actual vs predicted results for all models. 📈
        """
        try:
            plt.figure(figsize=(15, 8))
            plt.plot(
                self.test.index, self.test[self.column], label="Actual", linewidth=2
            )

            for model_name, predictions in self.prediction.items():
                plt.plot(
                    self.test.index,
                    predictions,
                    label=f"{model_name} Prediction",
                    linestyle="--",
                )

            plt.title("📈 Model Predictions Comparison")
            plt.xlabel("Date")
            plt.ylabel(self.column)
            plt.legend()
            plt.show()  # 🖼️ Display the plot
            self.logger.info("✅ Results plotted successfully")
        except Exception as e:
            # 🚨 Log the error details and raise an exception
            self.logger.error(f"⚠️ Error in plotting results: {e} ❌")
            raise ValueError("Plotting results failed")

    def save_best_model(self, model_name="LSTM"):
        """
        💾 Save the best model for future use. 🛠️
        """
        try:
            if model_name in self.model:
                model_data = self.model[model_name]
                if model_name == "LSTM":
                    # 💾 Save the LSTM model
                    model = model_data["model"]
                    model.save(f"../data/{model_name}_best_model.h5")
                    self.logger.info(f"✅ {model_name} model saved successfully.")
                else:
                    # 💾 Save the ARIMA or SARIMA model using joblib
                    joblib.dump(model_data, f"../data/{model_name}_best_model.pkl")
                    self.logger.info(f"✅ {model_name} model saved successfully.")
            else:
                self.logger.error(f"❌ {model_name} model not found for saving.")
        except Exception as e:
            # 🚨 Log the error details and raise an exception
            self.logger.error(f"⚠️ Error saving {model_name} model: {e}")
            raise ValueError(f"Model saving failed for {model_name}")