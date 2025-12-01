# Gaussian Mixture Models (GMM)

## 1. Concept
Gaussian Mixture Models (GMM) is a probabilistic model that assumes all the data points are generated from a mixture of a finite number of Gaussian distributions with unknown parameters.

Unlike K-Means, which performs **hard clustering** (a point belongs to exactly one cluster), GMM performs **soft clustering** (a point has a probability of belonging to each cluster).

### Key Differences from K-Means
| Feature | K-Means | GMM |
| :--- | :--- | :--- |
| **Cluster Shape** | Spherical (assumes equal variance) | Elliptical (flexible covariance) |
| **Assignment** | Hard (0 or 1) | Soft (Probabilities) |
| **Parameters** | Means ($\mu$) | Means ($\mu$), Covariances ($\Sigma$), Weights ($\pi$) |
| **Algorithm** | Coordinate Descent | Expectation-Maximization (EM) |

## 2. Mathematical Foundation

A GMM is defined by:
$$ P(x) = \sum_{k=1}^{K} \pi_k \mathcal{N}(x | \mu_k, \Sigma_k) $$

Where:
*   $\pi_k$: Mixing coefficient (weight) for cluster $k$, $\sum \pi_k = 1$.
*   $\mathcal{N}(x | \mu_k, \Sigma_k)$: Multivariate Gaussian density.

### Expectation-Maximization (EM) Algorithm

1.  **Initialization**: Randomly initialize $\mu_k, \Sigma_k, \pi_k$.
2.  **E-Step (Expectation)**: Calculate the "responsibility" $\gamma(z_{nk})$ (probability that point $n$ belongs to cluster $k$).
    $$ \gamma(z_{nk}) = \frac{\pi_k \mathcal{N}(x_n | \mu_k, \Sigma_k)}{\sum_{j=1}^{K} \pi_j \mathcal{N}(x_n | \mu_j, \Sigma_j)} $$
3.  **M-Step (Maximization)**: Update parameters using the current responsibilities.
    *   $N_k = \sum_{n=1}^{N} \gamma(z_{nk})$ (Effective number of points in cluster $k$)
    *   $\mu_k^{new} = \frac{1}{N_k} \sum_{n=1}^{N} \gamma(z_{nk}) x_n$
    *   $\Sigma_k^{new} = \frac{1}{N_k} \sum_{n=1}^{N} \gamma(z_{nk}) (x_n - \mu_k^{new})(x_n - \mu_k^{new})^T$
    *   $\pi_k^{new} = \frac{N_k}{N}$
4.  **Convergence**: Repeat E and M steps until the Log-Likelihood stabilizes.

## 3. Implementation Details
*   **Scratch**: Implements the EM algorithm manually using `numpy` and `scipy.stats.multivariate_normal`.
*   **Sklearn**: Uses `sklearn.mixture.GaussianMixture`. Demonstrates model selection using AIC/BIC.

## 4. How to Run
```bash
# Run scratch implementation
python 00_scratch.py

# Run sklearn implementation
python 01_sklearn.py
```
