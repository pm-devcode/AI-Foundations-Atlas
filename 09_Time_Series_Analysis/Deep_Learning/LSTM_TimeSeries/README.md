# LSTM for Time Series Forecasting

## 1. Introduction
While statistical models like ARIMA are powerful for linear, stationary data, Deep Learning models like Long Short-Term Memory (LSTM) networks excel at capturing complex, non-linear dependencies and long-term patterns in time series data.

## 2. LSTM vs. ARIMA
| Feature | ARIMA | LSTM |
| :--- | :--- | :--- |
| **Type** | Statistical (Linear) | Deep Learning (Non-Linear) |
| **Stationarity** | Required (via differencing) | Not strictly required (but helpful) |
| **Dependencies** | Short-term (lagged values) | Long-term (memory cell) |
| **Multivariate** | Difficult (ARIMAX) | Native support |
| **Data Size** | Works well with small data | Requires large data |

## 3. Real-World Analogy
*   **ARIMA (The Accountant):** Predicts next month's budget by looking at the exact numbers from the last 3 months and applying a fixed growth rate. "Last month was \$100, trend is +5%, so next is \$105."
*   **LSTM (The Trader):** Predicts the stock price by looking at the chart shape over the last year. It recognizes patterns like "head and shoulders" or "double bottom" that aren't just simple linear trends.

## 4. Implementation Details
*   **`01_lstm_ts.py`**: A PyTorch implementation of LSTM for forecasting a synthetic time series (Sine wave + Trend + Noise).
    *   **Data Preparation:** Sliding window approach (Sequence -> Next Value).
    *   **Normalization:** MinMax scaling (crucial for Neural Networks).
    *   **Model:** Single-layer LSTM followed by a Linear layer.
    *   **Result:** The model successfully learns both the upward trend and the cyclic sine wave pattern.

## 5. Applications
*   **Stock Price Prediction:** (Though very noisy).
*   **Energy Consumption:** Predicting grid load.
*   **Weather Forecasting:** Temperature, rainfall.
*   **Anomaly Detection:** Predicting normal behavior and flagging deviations.

## 6. Results

### Synthetic Data
![Data](assets/lstm_data.png)
*The input time series combining a linear trend, a sine wave, and random noise.*

### Forecast
![Forecast](assets/lstm_forecast.png)
*The LSTM's prediction on the test set (orange dashed line) closely follows the true values (green line), capturing both the trend and the seasonality.*

