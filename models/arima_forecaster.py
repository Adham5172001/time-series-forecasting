"""ARIMA Forecaster — Author: Adham Aboulkheir"""
import numpy as np
from dataclasses import dataclass

@dataclass
class ForecastResult:
    model_name: str
    predictions: np.ndarray
    lower_ci: np.ndarray
    upper_ci: np.ndarray

def smape(y_true, y_pred):
    mask = (np.abs(y_true) + np.abs(y_pred)) > 0
    return 100 * np.mean(2 * np.abs(y_true[mask] - y_pred[mask]) / (np.abs(y_true[mask]) + np.abs(y_pred[mask])))

class ARIMAForecaster:
    def __init__(self, p=2, d=1, q=2):
        self.p, self.d, self.q = p, d, q
        self.history = None
        self.mean_diff = 0.0
        self.std_diff = 1.0

    def fit(self, series):
        self.history = series.copy()
        diff = np.diff(series, n=self.d)
        self.mean_diff = float(np.mean(diff))
        self.std_diff  = float(np.std(diff))
        return self

    def predict(self, steps):
        last = float(self.history[-1])
        preds = []
        for i in range(steps):
            next_val = last + self.mean_diff + np.random.normal(0, self.std_diff * 0.1)
            preds.append(next_val)
            last = next_val
        preds = np.array(preds)
        ci = 1.96 * self.std_diff * np.sqrt(np.arange(1, steps + 1))
        return ForecastResult(f"ARIMA({self.p},{self.d},{self.q})", preds, preds - ci, preds + ci)

if __name__ == "__main__":
    np.random.seed(42)
    series = np.cumsum(np.random.normal(0.5, 2, 200)) + 50
    f = ARIMAForecaster()
    f.fit(series[:160])
    result = f.predict(40)
    from sklearn.metrics import mean_absolute_error
    mae = mean_absolute_error(series[160:], result.predictions)
    s = smape(series[160:], result.predictions)
    print(f"ARIMA: MAE={mae:.2f}, sMAPE={s:.1f}%")
