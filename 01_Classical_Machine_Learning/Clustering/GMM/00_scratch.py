import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
import os

# Ensure assets directory exists
os.makedirs('assets', exist_ok=True)

# --- 1. Data Generation ---
np.random.seed(42)

# Generate synthetic data from 3 Gaussian distributions
# Cluster 1
mean1 = [0, 0]
cov1 = [[1, 0.5], [0.5, 1]]
data1 = np.random.multivariate_normal(mean1, cov1, 200)

# Cluster 2
mean2 = [5, 5]
cov2 = [[1, -0.5], [-0.5, 1]]
data2 = np.random.multivariate_normal(mean2, cov2, 200)

# Cluster 3
mean3 = [0, 5]
cov3 = [[0.5, 0], [0, 0.5]]
data3 = np.random.multivariate_normal(mean3, cov3, 200)

X = np.vstack([data1, data2, data3])

# Visualization of Raw Data
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], s=10, alpha=0.6, color='gray')
plt.title("Synthetic Data (3 Gaussian Clusters)")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.savefig("assets/gmm_data.png")
print("Saved assets/gmm_data.png")

# --- 2. GMM Implementation (EM Algorithm) ---

class GMMScratch:
    def __init__(self, k=3, max_iter=100, tol=1e-4):
        self.k = k # Number of clusters
        self.max_iter = max_iter
        self.tol = tol
        
    def fit(self, X):
        n_samples, n_features = X.shape
        
        # --- Initialization ---
        # 1. Initialize means randomly (picking random points from data)
        self.means = X[np.random.choice(n_samples, self.k, replace=False)]
        
        # 2. Initialize covariances (Identity matrices)
        self.covariances = [np.eye(n_features) for _ in range(self.k)]
        
        # 3. Initialize mixing coefficients (pi) uniformly
        self.pi = np.ones(self.k) / self.k
        
        # Log-likelihood history
        self.log_likelihoods = []
        
        for i in range(self.max_iter):
            # --- E-Step (Expectation) ---
            # Calculate responsibilities (gamma): P(z_k | x_i)
            # gamma[i, k] = pi_k * N(x_i | mu_k, Sigma_k) / Sum_j(pi_j * N(x_i | mu_j, Sigma_j))
            
            responsibilities = np.zeros((n_samples, self.k))
            
            for k in range(self.k):
                # Use scipy's multivariate_normal for PDF calculation stability
                distribution = multivariate_normal(mean=self.means[k], cov=self.covariances[k])
                responsibilities[:, k] = self.pi[k] * distribution.pdf(X)
                
            # Normalize responsibilities (denominator of Bayes rule)
            # Sum over k for each sample
            total_responsibility = np.sum(responsibilities, axis=1, keepdims=True)
            responsibilities = responsibilities / total_responsibility
            
            # --- M-Step (Maximization) ---
            # Update parameters to maximize likelihood given responsibilities
            
            # N_k: Effective number of points in cluster k
            N_k = np.sum(responsibilities, axis=0)
            
            for k in range(self.k):
                # 1. Update Means
                # mu_k = (1/N_k) * Sum_i(gamma_ik * x_i)
                self.means[k] = np.sum(responsibilities[:, k].reshape(-1, 1) * X, axis=0) / N_k[k]
                
                # 2. Update Covariances
                # Sigma_k = (1/N_k) * Sum_i(gamma_ik * (x_i - mu_k)(x_i - mu_k)^T)
                diff = X - self.means[k] # (n_samples, n_features)
                
                # Weighted covariance calculation
                # We need to weight each sample's contribution to covariance by its responsibility
                weighted_diff = responsibilities[:, k].reshape(-1, 1) * diff
                self.covariances[k] = np.dot(weighted_diff.T, diff) / N_k[k]
                
                # Add small epsilon to diagonal for numerical stability
                self.covariances[k] += np.eye(n_features) * 1e-6
                
                # 3. Update Mixing Coefficients
                # pi_k = N_k / N
                self.pi[k] = N_k[k] / n_samples
                
            # --- Check Convergence (Log-Likelihood) ---
            log_likelihood = np.sum(np.log(total_responsibility))
            self.log_likelihoods.append(log_likelihood)
            
            if i > 0 and abs(log_likelihood - self.log_likelihoods[-2]) < self.tol:
                print(f"Converged at iteration {i}")
                break
                
        self.responsibilities = responsibilities
        
    def predict(self, X):
        # Hard assignment based on max responsibility
        probs = np.zeros((X.shape[0], self.k))
        for k in range(self.k):
            probs[:, k] = self.pi[k] * multivariate_normal(self.means[k], self.covariances[k]).pdf(X)
        return np.argmax(probs, axis=1)

# --- 3. Run Training ---
gmm = GMMScratch(k=3)
gmm.fit(X)
labels = gmm.predict(X)

# --- 4. Visualization ---
plt.figure(figsize=(12, 5))

# Plot 1: Clusters
plt.subplot(1, 2, 1)
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', s=10, alpha=0.6)
plt.scatter(gmm.means[:, 0], gmm.means[:, 1], c='red', marker='x', s=100, label='Centroids')

# Plot contours
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
grid_points = np.c_[xx.ravel(), yy.ravel()]

for k in range(3):
    z = multivariate_normal(gmm.means[k], gmm.covariances[k]).pdf(grid_points)
    z = z.reshape(xx.shape)
    plt.contour(xx, yy, z, levels=3, colors='black', alpha=0.3)

plt.title("GMM Clustering (Scratch)")
plt.legend()

# Plot 2: Log-Likelihood
plt.subplot(1, 2, 2)
plt.plot(gmm.log_likelihoods)
plt.title("Log-Likelihood Convergence")
plt.xlabel("Iteration")
plt.ylabel("Log-Likelihood")

plt.tight_layout()
plt.savefig("assets/gmm_scratch_result.png")
print("Saved assets/gmm_scratch_result.png")
