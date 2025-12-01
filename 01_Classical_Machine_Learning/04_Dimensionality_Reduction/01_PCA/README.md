# Principal Component Analysis (PCA)

## 1. Executive Summary
Principal Component Analysis (PCA) is a linear dimensionality reduction technique. It transforms the data into a new coordinate system where the axes (Principal Components) are ordered by the amount of variance they explain. The goal is to reduce the number of variables (features) while preserving as much information (variance) as possible.

## 2. Historical Context
PCA was invented by **Karl Pearson** in **1901** as an analogue of the principal axis theorem in mechanics. It was later independently developed and named by **Harold Hotelling** in the 1930s. It is one of the oldest and most widely used techniques in multivariate analysis.

## 3. Real-World Analogy
Think of **Taking a Photo of a Teapot**.
*   The teapot is a 3D object (3 dimensions).
*   A photo is a 2D projection (2 dimensions).
*   You want to choose the angle that captures the most information about the teapot's shape (handle, spout, lid).
*   If you take a photo from the top, it might just look like a circle (bad projection, low variance).
*   If you take it from the side, you see the unique shape (good projection, high variance).
*   **PCA** automatically finds the "best angle" (Principal Components) to project the data onto.

## 4. Key Concepts

1.  **Variance**: A measure of the spread of data. PCA assumes that features with higher variance contain more information.
2.  **Covariance**: Measures how two variables change together.
3.  **Eigenvectors**: The directions of the axes of the new coordinate system (Principal Components).
4.  **Eigenvalues**: The magnitude of variance in the direction of the corresponding eigenvector.

## 5. Mathematics

### 1. Covariance Matrix
First, we center the data (subtract mean). Then we compute the covariance matrix $\Sigma$:
$$ \Sigma = \frac{1}{n-1} X^T X $$
Where $X$ is the centered data matrix ($n \times d$).

### 2. Eigendecomposition
We solve for eigenvalues $\lambda$ and eigenvectors $v$:
$$ \Sigma v = \lambda v $$

### 3. Projection
We sort the eigenvectors by eigenvalues in descending order. We pick the top $k$ eigenvectors to form a matrix $W$.
The new data $Z$ is obtained by:
$$ Z = X W $$

## 6. Implementation Details

1.  **`00_scratch.py`**: Implementation using `numpy.linalg.eig`.
    *   Standardize data.
    *   Compute Covariance Matrix.
    *   Compute and sort Eigenpairs.
    *   Project data.
2.  **`01_sklearn.py`**: Reference implementation using `sklearn.decomposition.PCA`.

## 7. Results

### Scratch Implementation
![Scratch PCA](assets/scratch_pca.png)

*Left: Original data with red arrows showing the Principal Components. Right: Data projected onto the first Principal Component (1D).*

### Sklearn Implementation
![Sklearn PCA](assets/sklearn_pca.png)

*Data rotated into the new coordinate system defined by the Principal Components.*

## 8. How to Run

```bash
python 00_scratch.py
python 01_sklearn.py
```
