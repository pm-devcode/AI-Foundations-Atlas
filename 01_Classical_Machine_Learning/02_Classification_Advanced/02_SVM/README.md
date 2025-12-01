# Support Vector Machines (SVM)

## 1. Executive Summary
Support Vector Machine (SVM) is a powerful supervised learning algorithm used for classification and regression. Unlike other classifiers that might just find *any* boundary separating classes, SVM finds the *optimal* hyperplane that maximizes the margin (distance) between the nearest data points of different classes. This "wide margin" approach makes SVM robust and effective, especially in high-dimensional spaces.

## 2. Historical Context
The foundations of SVM were laid by **Vladimir Vapnik** and **Alexey Chervonenkis** in 1963 with the "Generalized Portrait" algorithm. However, the modern version of SVM, which introduced the **Kernel Trick** to handle non-linear data, was developed by Vapnik and his colleagues (Boser and Guyon) at AT&T Bell Labs in **1992**. For a long time, SVM was considered the state-of-the-art algorithm for many tasks (like handwriting recognition) before the Deep Learning resurgence in 2012.

## 3. Real-World Analogy
Imagine a road separating two groups of houses (Class A and Class B).
*   You could draw many lines (roads) to separate them.
*   However, the safest road is the widest one—the one that keeps the maximum distance from the houses on both sides.
*   **SVM** tries to build this "widest possible road" (maximum margin).
*   The houses closest to the road are the **Support Vectors**—if you move other houses, the road doesn't change, but if you move these specific houses, the road's position shifts.

## 4. Key Concepts

1.  **Hyperplane**: The decision boundary. In 2D, it's a line; in 3D, a plane.
    $$ w \cdot x - b = 0 $$
2.  **Margin**: The distance between the hyperplane and the nearest data points. SVM maximizes this.
3.  **Support Vectors**: The data points closest to the hyperplane. They define the margin.
4.  **Kernel Trick**: A method to map data into a higher-dimensional space where it becomes linearly separable (e.g., using RBF kernel).

## 5. Mathematics (Linear SVM - Soft Margin)

We want to minimize the norm of the weight vector $||w||$ (maximizing margin) while penalizing misclassifications.

**Cost Function (Hinge Loss + Regularization):**

$$ J(w, b) = \frac{1}{2} ||w||^2 + C \sum_{i=1}^{n} \max(0, 1 - y_i(w \cdot x_i - b)) $$

Where:
*   $y_i \in \{-1, 1\}$ are class labels.
*   $C$ is the regularization parameter (balance between margin width and classification errors).

**Gradient Descent Update Rule:**

For each example $i$:
If $y_i(w \cdot x_i - b) \ge 1$ (correctly classified with margin):
$$ w = w - \alpha (2\lambda w) $$
If $y_i(w \cdot x_i - b) < 1$ (error or inside margin):
$$ w = w - \alpha (2\lambda w - y_i x_i) $$
$$ b = b - \alpha (y_i) $$

*(Note: In the scratch implementation, we use $\lambda = 1/C$)*

## 6. Implementation Details

1.  **`00_scratch.py`**: Implementation of Linear SVM using Gradient Descent minimizing Hinge Loss.
2.  **`01_sklearn.py`**: Comparison of Linear SVM and Non-linear SVM (RBF Kernel) using `scikit-learn`.

## 7. Results

### Scratch Implementation (Linear SVM)
![Scratch SVM](assets/scratch_svm_boundary.png)

### Sklearn Implementation (Linear vs RBF)
![Sklearn SVM](assets/sklearn_svm_comparison.png)

## 8. How to Run

```bash
python 00_scratch.py
python 01_sklearn.py
```
