"""Unified Time Series Forecasting — Author: Adham Aboulkheir"""
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from models.arima_forecaster import ARIMAForecaster, smape
from models.lstm_forecaster import LSTMForecaster
from sklearn.metrics import mean_absolute_error

class TimeSeriesForecaster:
    MODELS = {
        "arima": lambda: ARIMAForecaster(p=2, d=1, q=2),
        "lstm":  lambda: LSTMForecaster(hidden_size=64, lookback=30),
        "tft":   lambda: LSTMForecaster(hidden_size=128, n_layers=3, lookback=40),
    }
    def __init__(self, model="lstm", horizon=30):
        self.model_name = model
        self.horizon = horizon
        self.model = self.MODELS.get(model, self.MODELS["lstm"])()

    def fit(self, series):
        self.series = series
        self.model.fit(series)
        return self

    def predict(self):
        return self.model.predict(self.horizon)

    def evaluate(self, y_true):
        result = self.predict()
        n = min(len(y_true), len(result.predictions))
        return {"model": self.model_name,
                "mae": float(mean_absolute_error(y_true[:n], result.predictions[:n])),
                "smape": float(smape(y_true[:n], result.predictions[:n]))}

if __name__ == "__main__":
    np.random.seed(42)
    t = np.arange(200)
    series = 50 + 10*np.sin(2*np.pi*t/52) + 0.05*t + np.random.normal(0, 2, 200)
    train, test = series[:160], series[160:]
    print("Time Series Forecasting Comparison:")
    for name in ["arima", "lstm", "tft"]:
        f = TimeSeriesForecaster(model=name, horizon=40)
        f.fit(train)
        m = f.evaluate(test)
        print(f"  {name.upper():6s}: MAE={m['mae']:.2f}, sMAPE={m['smape']:.1f}%")
