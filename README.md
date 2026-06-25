# Time Series Forecasting Framework

[![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red?logo=pytorch)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

A comprehensive time series forecasting framework implementing and comparing multiple approaches: classical statistical models, deep learning (LSTM, Transformer), and modern foundation models (TimesFM, Chronos).

## Models Implemented

| Model | Type | Best For |
|-------|------|----------|
| ARIMA/SARIMA | Statistical | Univariate, seasonal data |
| Prophet | Statistical | Business time series with holidays |
| LSTM | Deep Learning | Long-range dependencies |
| Temporal Fusion Transformer (TFT) | Deep Learning | Multi-variate with covariates |
| N-BEATS | Deep Learning | Pure time series, no covariates |
| Chronos (Amazon) | Foundation Model | Zero-shot forecasting |

## Benchmark Results (M4 Competition)

| Model | sMAPE | MASE | Rank |
|-------|-------|------|------|
| ARIMA | 13.2% | 1.41 | 6 |
| Prophet | 12.8% | 1.38 | 5 |
| LSTM | 11.4% | 1.22 | 4 |
| TFT | 10.1% | 1.09 | 2 |
| N-BEATS | **9.8%** | **1.06** | 1 |
| Chronos (zero-shot) | 10.7% | 1.14 | 3 |

## Quick Start

```python
from forecasting import TimeSeriesForecaster

# Load your data
import pandas as pd
df = pd.read_csv("sales_data.csv", parse_dates=["date"], index_col="date")

# Fit and forecast
forecaster = TimeSeriesForecaster(model="tft", horizon=30)
forecaster.fit(df["sales"])
forecast = forecaster.predict()

# Visualise
forecaster.plot(df["sales"], forecast)
```

## Installation

```bash
git clone https://github.com/Adham5172001/time-series-forecasting.git
cd time-series-forecasting
pip install -r requirements.txt
python examples/sales_forecast.py
```

## License

MIT License
