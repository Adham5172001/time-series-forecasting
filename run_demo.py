"""Time Series Forecasting Demo — Author: Adham Aboulkheir"""
import numpy as np, matplotlib, os, sys
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(__file__))
from forecasting import TimeSeriesForecaster

def main():
    print("Time Series Forecasting Framework Demo")
    os.makedirs("outputs", exist_ok=True)
    np.random.seed(42)
    t = np.arange(200)
    series = 50 + 10*np.sin(2*np.pi*t/52) + 5*np.sin(2*np.pi*t/12) + 0.05*t + np.random.normal(0, 2, 200)
    train, test = series[:160], series[160:]
    results = {}
    for name in ["arima", "lstm", "tft"]:
        f = TimeSeriesForecaster(model=name, horizon=40)
        f.fit(train)
        metrics = f.evaluate(test)
        results[name] = (metrics, f.predict())
        print(f"  {name.upper():6s}: MAE={metrics['mae']:.2f}, sMAPE={metrics['smape']:.1f}%")
    fig, ax = plt.subplots(figsize=(12, 5), facecolor="#0d1117")
    ax.set_facecolor("#161b22")
    ax.plot(t[:160], train, color="#58a6ff", linewidth=1.5, label="Historical", alpha=0.8)
    ax.plot(t[160:], test, color="#3fb950", linewidth=2, label="Actual")
    colors = {"arima": "#ff7b72", "lstm": "#00c9b1", "tft": "#f4a261"}
    for name, (metrics, result) in results.items():
        ax.plot(t[160:], result.predictions, color=colors[name], linewidth=1.5, linestyle="--",
                label=f"{name.upper()} (sMAPE={metrics['smape']:.1f}%)")
    ax.axvline(x=160, color="white", linestyle=":", alpha=0.4)
    ax.set_title("Multi-Model Time Series Forecast", color="white")
    ax.set_xlabel("Time Step", color="white")
    ax.set_ylabel("Value", color="white")
    ax.legend(facecolor="#161b22", labelcolor="white", fontsize=8)
    ax.tick_params(colors="white")
    ax.grid(alpha=0.3, color="#21262d")
    plt.tight_layout()
    plt.savefig("outputs/time_series_results.png", dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print("  Saved: outputs/time_series_results.png")

if __name__ == "__main__":
    main()
