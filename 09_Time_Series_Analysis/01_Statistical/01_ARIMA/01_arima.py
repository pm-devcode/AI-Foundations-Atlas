import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import os

# Ensure assets directory exists
os.makedirs('assets', exist_ok=True)

# --- 1. Data Generation ---
# Simulate a time series with Trend + Noise (ARIMA is good for this)
np.random.seed(42)
n_samples = 200
t = np.arange(n_samples)

# Linear Trend
trend = 0.5 * t

# Autoregressive component (AR(1))
# y_t = 0.7 * y_{t-1} + noise
ar_component = np.zeros(n_samples)
for i in range(1, n_samples):
    ar_component[i] = 0.7 * ar_component[i-1] + np.random.normal(0, 5)

# Combine
y = trend + ar_component

# Create DataFrame
df = pd.DataFrame({'value': y}, index=pd.date_range(start='2023-01-01', periods=n_samples, freq='D'))

print("Data Head:")
print(df.head())

# --- 2. Visualization & Analysis ---
plt.figure(figsize=(10, 6))
plt.plot(df['value'])
plt.title("Synthetic Time Series (Trend + AR(1))")
plt.xlabel("Date")
plt.ylabel("Value")
plt.savefig("assets/arima_data.png")
print("Saved assets/arima_data.png")

# ACF and PACF
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
plot_acf(df['value'], ax=ax1)
plot_pacf(df['value'], ax=ax2)
plt.tight_layout()
plt.savefig("assets/arima_acf_pacf.png")
print("Saved assets/arima_acf_pacf.png")

# --- 3. ARIMA Model ---
# We see a trend, so d=1 (Differencing) is likely needed.
# We generated AR(1), so p=1 might be good.
# Let's try ARIMA(1, 1, 1)
model = ARIMA(df['value'], order=(1, 1, 1))
model_fit = model.fit()

print(model_fit.summary())

# --- 4. Forecasting ---
forecast_steps = 30
forecast_res = model_fit.get_forecast(steps=forecast_steps)
forecast_mean = forecast_res.predicted_mean
conf_int = forecast_res.conf_int()

# Plot Forecast
plt.figure(figsize=(12, 6))
plt.plot(df.index, df['value'], label='Observed')
plt.plot(forecast_mean.index, forecast_mean, label='Forecast', color='red')
plt.fill_between(conf_int.index, conf_int.iloc[:, 0], conf_int.iloc[:, 1], color='pink', alpha=0.3)
plt.title("ARIMA(1,1,1) Forecast")
plt.legend()
plt.savefig("assets/arima_forecast.png")
print("Saved assets/arima_forecast.png")

# --- 5. Diagnostics ---
model_fit.plot_diagnostics(figsize=(10, 8))
plt.tight_layout()
plt.savefig("assets/arima_diagnostics.png")
print("Saved assets/arima_diagnostics.png")
