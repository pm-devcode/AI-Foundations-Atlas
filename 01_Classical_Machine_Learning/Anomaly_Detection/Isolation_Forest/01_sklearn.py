import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
import os

# Ensure assets directory exists
os.makedirs('assets', exist_ok=True)

# --- 1. Data Generation ---
rng = np.random.RandomState(42)

# Generate train data (regular observations)
X = 0.3 * rng.randn(100, 2)
X_train = np.r_[X + 2, X - 2]

# Generate some regular novel observations (test set)
X = 0.3 * rng.randn(20, 2)
X_test = np.r_[X + 2, X - 2]

# Generate some abnormal novel observations (outliers)
X_outliers = rng.uniform(low=-4, high=4, size=(20, 2))

# --- 2. Scikit-Learn Isolation Forest ---
# contamination: proportion of outliers in the data set. 'auto' determines it automatically.
clf = IsolationForest(max_samples=100, random_state=rng, contamination=0.1)
clf.fit(X_train)

# Predict: 1 for inliers, -1 for outliers
y_pred_train = clf.predict(X_train)
y_pred_test = clf.predict(X_test)
y_pred_outliers = clf.predict(X_outliers)

# --- 3. Visualization ---
plt.figure(figsize=(10, 6))

xx, yy = np.meshgrid(np.linspace(-5, 5, 50), np.linspace(-5, 5, 50))
Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.title("Isolation Forest (Scikit-Learn)")
plt.contourf(xx, yy, Z, cmap=plt.cm.Blues_r)

b1 = plt.scatter(X_train[:, 0], X_train[:, 1], c='white', s=20, edgecolor='k')
b2 = plt.scatter(X_test[:, 0], X_test[:, 1], c='green', s=20, edgecolor='k')
c = plt.scatter(X_outliers[:, 0], X_outliers[:, 1], c='red', s=20, edgecolor='k')

plt.axis('tight')
plt.xlim((-5, 5))
plt.ylim((-5, 5))
plt.legend([b1, b2, c],
           ["Training observations",
            "New regular observations",
            "New abnormal observations"],
           loc="upper left")

plt.savefig("assets/if_sklearn_result.png")
print("Saved assets/if_sklearn_result.png")
