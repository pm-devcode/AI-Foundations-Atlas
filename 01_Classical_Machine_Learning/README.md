# Classical Machine Learning

## Overview
Classical Machine Learning refers to the generation of algorithms developed largely before the "Deep Learning revolution" (pre-2012). These methods are rooted heavily in statistics, linear algebra, and optimization theory. They remain the industry standard for tabular data and scenarios requiring high interpretability.

## Key Categories

### 1. Regression
Predicting continuous values.
*   **Linear Regression**: The "Hello World" of ML. Fitting a line to data.
*   **Logistic Regression**: Despite the name, used for binary classification.

### 2. Classification
Predicting discrete labels.
*   **KNN (K-Nearest Neighbors)**: Instance-based learning. "Tell me who your friends are, and I'll tell you who you are."
*   **SVM (Support Vector Machines)**: Finding the optimal hyperplane that maximizes the margin between classes.
*   **Naive Bayes**: Probabilistic classifiers based on Bayes' theorem with strong independence assumptions.

### 3. Clustering
Unsupervised grouping of data.
*   **K-Means**: Partitioning data into K distinct clusters based on distance to centroids.
*   **DBSCAN**: Density-based clustering that can find arbitrarily shaped clusters and outliers.
*   **GMM (Gaussian Mixture Models)**: Probabilistic clustering assuming data is generated from a mixture of Gaussian distributions.

### 4. Dimensionality Reduction
Reducing the number of random variables under consideration.
*   **PCA (Principal Component Analysis)**: Projecting data onto orthogonal axes of maximum variance.
*   **LDA (Linear Discriminant Analysis)**: Finding a linear combination of features that characterizes or separates two or more classes.

### 5. Ensemble Methods
Combining multiple models to improve performance.
*   **Decision Trees**: Flowchart-like structures for decision making.
*   **Random Forests**: Bagging method using many decorrelated trees.
*   **Gradient Boosting**: Boosting method that builds trees sequentially to correct errors of previous trees.

### 6. Anomaly Detection
Identifying rare items, events or observations.
*   **Isolation Forest**: Explicitly isolates anomalies rather than profiling normal data points.
