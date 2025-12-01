import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
from sklearn.ensemble import GradientBoostingRegressor

# Ensure assets directory exists
script_dir = os.path.dirname(os.path.abspath(__file__))
assets_dir = os.path.join(script_dir, "assets")
os.makedirs(assets_dir, exist_ok=True)

# Generate synthetic data (Sine wave) - Same as scratch
np.random.seed(42)
X = np.sort(5 * np.random.rand(80, 1), axis=0)
y = np.sin(X).ravel()
y[::5] += 3 * (0.5 - np.random.rand(16)) # Add noise

# Train Sklearn Model
gb = GradientBoostingRegressor(n_estimators=50, learning_rate=0.1, max_depth=2, random_state=42)
gb.fit(X, y)

# Predict
X_test = np.arange(0.0, 5.0, 0.01)[:, np.newaxis]
y_pred = gb.predict(X_test)

# Visualization
plt.figure(figsize=(10, 6))
plt.scatter(X, y, s=20, edgecolor="black", c="darkorange", label="data")
plt.plot(X_test, y_pred, color="cornflowerblue", label="Gradient Boosting Prediction", linewidth=2)
plt.xlabel("data")
plt.ylabel("target")
plt.title("Gradient Boosting Regression (Scikit-Learn Reference)")
plt.legend()

output_path = os.path.join(assets_dir, "sklearn_gb_regression.png")
plt.savefig(output_path)
print(f"Saved visualization to {output_path}")
