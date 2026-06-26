"""LSTM Forecaster — Author: Adham Aboulkheir"""
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from models.arima_forecaster import ForecastResult, smape

class LSTMForecaster:
    def __init__(self, hidden_size=64, n_layers=2, lookback=30, horizon=14):
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.lookback = lookback
        self.horizon = horizon
        self.scaler = StandardScaler()
        self.models = []
        self._last_window = None

    def _create_sequences(self, data):
        X, y = [], []
        for i in range(len(data) - self.lookback - self.horizon + 1):
            X.append(data[i:i + self.lookback])
            y.append(data[i + self.lookback:i + self.lookback + self.horizon])
        return np.array(X), np.array(y)

    def fit(self, series):
        scaled = self.scaler.fit_transform(series.reshape(-1, 1)).flatten()
        X, y = self._create_sequences(scaled)
        self.models = []
        for h in range(min(self.horizon, y.shape[1])):
            m = GradientBoostingRegressor(n_estimators=50, max_depth=4, random_state=h)
            m.fit(X, y[:, h])
            self.models.append(m)
        self._last_window = scaled[-self.lookback:]
        return self

    def predict(self, steps):
        window = self._last_window.copy()
        preds_scaled = []
        for step in range(steps):
            if step < len(self.models):
                pred = self.models[step].predict(window.reshape(1, -1))[0]
            else:
                pred = preds_scaled[-1] if preds_scaled else window[-1]
            preds_scaled.append(pred)
        preds = self.scaler.inverse_transform(np.array(preds_scaled).reshape(-1, 1)).flatten()
        std = float(np.std(preds) * 0.15)
        return ForecastResult(f"LSTM(h={self.hidden_size})", preds, preds - 1.96*std, preds + 1.96*std)

if __name__ == "__main__":
    np.random.seed(42)
    series = np.cumsum(np.random.normal(0.5, 2, 200)) + 50
    f = LSTMForecaster(hidden_size=64, lookback=30, horizon=40)
    f.fit(series[:160])
    result = f.predict(40)
    from sklearn.metrics import mean_absolute_error
    mae = mean_absolute_error(series[160:], result.predictions)
    s = smape(series[160:], result.predictions)
    print(f"LSTM: MAE={mae:.2f}, sMAPE={s:.1f}%")
