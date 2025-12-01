# Decision Trees

## 1. Executive Summary
Decision Trees are a versatile supervised learning method used for both classification and regression. The model predicts the value of a target variable by learning simple decision rules inferred from the data features. It mimics human decision-making by breaking down a complex decision into a series of simpler choices.

## 2. Historical Context
The development of Decision Trees is largely attributed to **Leo Breiman** and colleagues who introduced **CART** (Classification and Regression Trees) in **1984**. Around the same time, **Ross Quinlan** developed **ID3** (1986) and later **C4.5** (1993), which became standard algorithms for generating decision trees. These methods revolutionized machine learning by providing interpretable models ("white box") in contrast to "black box" models like neural networks.

## 3. Real-World Analogy
Think of the game **"20 Questions"**.
*   You want to guess what object someone is thinking of.
*   You ask: "Is it alive?" (Yes/No).
*   If Yes: "Is it an animal?" (Yes/No).
*   If Yes: "Does it bark?" (Yes/No).
*   Each question splits the possibilities into smaller groups.
*   A Decision Tree does exactly this: it asks the most informative questions first to narrow down the answer as quickly as possible.

## 4. Key Concepts

1.  **Root Node**: Represents the entire population or sample.
2.  **Splitting**: Dividing a node into two or more sub-nodes.
3.  **Decision Node**: When a sub-node splits into further sub-nodes.
4.  **Leaf (Terminal) Node**: Nodes that do not split (contain the final prediction).

## 5. Mathematics (Splitting Criteria)

To decide the best split, we use metrics to measure "purity".

### 1. Entropy
Measures the disorder or impurity in a dataset.
$$ H(S) = - \sum_{i=1}^{c} p_i \log_2 p_i $$
Where $p_i$ is the probability of class $i$.

### 2. Information Gain
Measures the reduction in entropy after splitting dataset $S$ on attribute $A$.
$$ IG(S, A) = H(S) - \sum_{v \in Values(A)} \frac{|S_v|}{|S|} H(S_v) $$

### 3. Gini Impurity
An alternative to entropy (often computationally faster). Measures the probability of misclassifying a randomly chosen element.
$$ Gini = 1 - \sum_{i=1}^{c} (p_i)^2 $$

## 6. Implementation Details

1.  **`00_scratch.py`**: Custom implementation of the tree building algorithm (recursive splitting).
2.  **`01_sklearn.py`**: Reference implementation using `scikit-learn`.

## 7. Results

### Scratch Implementation (Decision Boundary)
![Scratch Decision Tree](assets/scratch_tree_boundary.png)

### Sklearn Implementation (Decision Boundary)
![Sklearn Decision Tree](assets/sklearn_tree_boundary.png)

## 8. How to Run

```bash
python 00_scratch.py
python 01_sklearn.py
```
