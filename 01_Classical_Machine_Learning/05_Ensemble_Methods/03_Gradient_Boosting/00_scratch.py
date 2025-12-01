import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
from sklearn.tree import DecisionTreeRegressor

# Ensure assets directory exists
script_dir = os.path.dirname(os.path.abspath(__file__))
assets_dir = os.path.join(script_dir, "assets")
os.makedirs(assets_dir, exist_ok=True)

class GradientBoostingRegressorScratch:
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.trees = []
        self.initial_prediction = None

    def fit(self, X, y):
        # 1. Initialize with mean
        self.initial_prediction = np.mean(y)
        F_m = np.full(y.shape, self.initial_prediction)
        
        for _ in range(self.n_estimators):
            # 2. Calculate residuals
            residuals = y - F_m
            
            # 3. Fit a weak learner to residuals
            tree = DecisionTreeRegressor(max_depth=self.max_depth)
            tree.fit(X, residuals)
            self.trees.append(tree)
            
            # 4. Update model prediction
            update = tree.predict(X)
            F_m += self.learning_rate * update

    def predict(self, X):
        # Start with initial prediction
        F_m = np.full(X.shape[0], self.initial_prediction)
        
        # Add contributions from all trees
        for tree in self.trees:
            F_m += self.learning_rate * tree.predict(X)
            
        return F_m

# --- Data Generation & Visualization ---
if __name__ == "__main__":
    # Generate synthetic data (Sine wave)
    np.random.seed(42)
    X = np.sort(5 * np.random.rand(80, 1), axis=0)
    y = np.sin(X).ravel()
    y[::5] += 3 * (0.5 - np.random.rand(16)) # Add noise
    
    # Train Scratch Model
    gb = GradientBoostingRegressorScratch(n_estimators=50, learning_rate=0.1, max_depth=2)
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
    plt.title("Gradient Boosting Regression (Scratch Implementation)")
    plt.legend()
    
    output_path = os.path.join(assets_dir, "scratch_gb_regression.png")
    plt.savefig(output_path)
    print(f"Saved visualization to {output_path}")
