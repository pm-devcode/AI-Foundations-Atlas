# Isolation Forest

## 1. Concept
Isolation Forest is an unsupervised learning algorithm for anomaly detection. Unlike other methods that try to model the "normal" data (like GMM or One-Class SVM), Isolation Forest explicitly attempts to **isolate** anomalies.

### Core Idea
*   **Anomalies are few and different.**
*   If we randomly partition the data space, anomalies are likely to be isolated in fewer steps (shorter path length in a tree) than normal points.
*   Normal points are clustered together and require more cuts to be isolated (deeper in the tree).

## 2. Mathematical Foundation

### Isolation Tree (iTree)
A binary tree built by recursively:
1.  Randomly selecting a feature.
2.  Randomly selecting a split value between the max and min of that feature.

### Anomaly Score
The score $s(x, n)$ for an instance $x$ is defined as:

$$ s(x, n) = 2^{-\frac{E(h(x))}{c(n)}} $$

Where:
*   $h(x)$: Path length of point $x$ (number of edges from root to leaf).
*   $E(h(x))$: Average path length across a forest of trees.
*   $c(n)$: Average path length of an unsuccessful search in a Binary Search Tree (BST) with $n$ nodes (used for normalization).

**Interpretation**:
*   $s \to 1$: High probability of being an anomaly (Short path).
*   $s \to 0$: Normal observation (Long path).
*   $s \approx 0.5$: No distinct anomaly.

## 3. Implementation Details
*   **Scratch**: Implements `IsolationTree` and `IsolationForest` classes. Calculates path lengths and anomaly scores manually.
*   **Sklearn**: Uses `sklearn.ensemble.IsolationForest`.

## 4. How to Run
```bash
# Run scratch implementation
python 00_scratch.py

# Run sklearn implementation
python 01_sklearn.py
```
