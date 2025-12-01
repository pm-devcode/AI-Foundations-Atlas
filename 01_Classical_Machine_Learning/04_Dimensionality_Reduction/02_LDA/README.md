# Linear Discriminant Analysis (LDA)

## 1. Concept
Linear Discriminant Analysis (LDA) is a **supervised** dimensionality reduction technique. Unlike PCA, which tries to maximize the variance of the data (ignoring class labels), LDA tries to maximize the **separability** between classes.

### Goal
Find a linear combination of features that characterizes or separates two or more classes of objects or events.

### PCA vs LDA
*   **PCA (Unsupervised)**: Finds axes of maximum variance. Good for compression.
*   **LDA (Supervised)**: Finds axes that maximize the distance between class means and minimize the spread within each class. Good for classification preprocessing.

## 2. Mathematical Foundation

LDA seeks a projection matrix $W$ that maximizes the Fisher's Criterion:

$$ J(W) = \frac{|W^T S_B W|}{|W^T S_W W|} $$

Where:
*   **$S_B$ (Between-Class Scatter)**: Measures how far the class means are from the overall mean.
    $$ S_B = \sum_{c} N_c (\mu_c - \mu)(\mu_c - \mu)^T $$
*   **$S_W$ (Within-Class Scatter)**: Measures how spread out the data is within each class.
    $$ S_W = \sum_{c} \sum_{x \in D_c} (x - \mu_c)(x - \mu_c)^T $$

The solution is given by the eigenvectors of $S_W^{-1} S_B$ corresponding to the largest eigenvalues.

## 3. Implementation Details
*   **Scratch**: Implements the calculation of $S_W$, $S_B$, and solves the generalized eigenvalue problem using `numpy.linalg.eig`.
*   **Sklearn**: Uses `sklearn.discriminant_analysis.LinearDiscriminantAnalysis`. Compares results with PCA.

## 4. How to Run
```bash
# Run scratch implementation
python 00_scratch.py

# Run sklearn implementation (LDA vs PCA)
python 01_sklearn.py
```
