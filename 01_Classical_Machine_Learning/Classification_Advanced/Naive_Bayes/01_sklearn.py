import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import os

# Ensure assets directory exists
os.makedirs('assets', exist_ok=True)

# --- Data Generation ---
X, y = make_classification(
    n_samples=300, n_features=2, n_informative=2, n_redundant=0,
    n_clusters_per_class=1, random_state=42, class_sep=1.5
)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Sklearn Implementation ---
# GaussianNB implements the Gaussian Naive Bayes algorithm for classification.
model = GaussianNB()
model.fit(X_train, y_train)
predictions = model.predict(X_test)

acc = accuracy_score(y_test, predictions)
print(f"Sklearn GaussianNB Accuracy: {acc:.4f}")

# --- Visualization ---
def plot_decision_boundary(X, y, model, title, filename):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                         np.arange(y_min, y_max, 0.1))
    
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, alpha=0.4, cmap='viridis')
    plt.scatter(X[:, 0], X[:, 1], c=y, s=20, edgecolor='k', cmap='viridis')
    plt.title(title)
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.savefig(filename)
    plt.close()
    print(f"Saved {filename}")

plot_decision_boundary(X_test, y_test, model, 
                      f"Sklearn GaussianNB - Acc: {acc:.2f}", 
                      "assets/sklearn_nb_boundary.png")
