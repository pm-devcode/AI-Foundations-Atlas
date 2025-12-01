import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import os

# Ensure assets directory exists
os.makedirs('assets', exist_ok=True)

class GaussianNaiveBayesScratch:
    def __init__(self):
        self.classes = None
        self.mean = {}
        self.var = {}
        self.priors = {}

    def fit(self, X, y):
        """
        Train the model by calculating mean, variance, and prior probability for each class.
        """
        n_samples, n_features = X.shape
        self.classes = np.unique(y)
        
        for c in self.classes:
            # Filter data for class c
            X_c = X[y == c]
            
            # Calculate statistics
            # Mean: Center of the Gaussian for this class
            self.mean[c] = np.mean(X_c, axis=0)
            # Variance: Spread of the Gaussian for this class
            self.var[c] = np.var(X_c, axis=0)
            # Prior: P(y) = count(c) / total_samples
            self.priors[c] = X_c.shape[0] / n_samples

    def _gaussian_pdf(self, class_idx, x):
        """
        Calculate the probability density of x given the class parameters (mean, var).
        P(x_i | y) = (1 / sqrt(2 * pi * var)) * exp(-(x - mean)^2 / (2 * var))
        """
        mean = self.mean[class_idx]
        var = self.var[class_idx]
        numerator = np.exp(-((x - mean) ** 2) / (2 * var))
        denominator = np.sqrt(2 * np.pi * var)
        return numerator / denominator

    def predict(self, X):
        """
        Predict class labels for samples in X.
        """
        y_pred = [self._predict_single(x) for x in X]
        return np.array(y_pred)

    def _predict_single(self, x):
        """
        Predict class for a single sample.
        We calculate the posterior probability P(y|x) proportional to P(x|y) * P(y).
        We work with log probabilities to avoid numerical underflow.
        """
        posteriors = []

        for c in self.classes:
            # Log Prior: log(P(y))
            prior = np.log(self.priors[c])
            
            # Log Likelihood: sum(log(P(x_i | y)))
            # We assume features are independent (Naive assumption)
            posterior = np.sum(np.log(self._gaussian_pdf(c, x)))
            
            # Posterior = Prior + Likelihood (in log space)
            posterior = prior + posterior
            posteriors.append(posterior)

        # Return class with highest posterior probability
        return self.classes[np.argmax(posteriors)]

# --- Data Generation ---
# Create a simple 2D dataset for visualization
X, y = make_classification(
    n_samples=300, n_features=2, n_informative=2, n_redundant=0,
    n_clusters_per_class=1, random_state=42, class_sep=1.5
)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Training ---
nb = GaussianNaiveBayesScratch()
nb.fit(X_train, y_train)
predictions = nb.predict(X_test)

accuracy = np.mean(predictions == y_test)
print(f"Scratch Naive Bayes Accuracy: {accuracy:.4f}")

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

plot_decision_boundary(X_test, y_test, nb, 
                      f"Gaussian Naive Bayes (Scratch) - Acc: {accuracy:.2f}", 
                      "assets/scratch_nb_boundary.png")
