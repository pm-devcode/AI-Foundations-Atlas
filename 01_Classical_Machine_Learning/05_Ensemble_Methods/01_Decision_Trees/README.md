# Decision Trees

## 1. Teoria (Theory)

**Drzewa Decyzyjne (Decision Trees)** to metoda uczenia nadzorowanego używana zarówno do klasyfikacji, jak i regresji. Model ten uczy się prostych reguł decyzyjnych wywnioskowanych z cech danych (features).

### Kluczowe koncepcje:

1.  **Węzeł korzenia (Root Node)**: Reprezentuje całą populację lub próbkę.
2.  **Podział (Splitting)**: Proces podziału węzła na dwa lub więcej pod-węzłów.
3.  **Węzeł decyzyjny (Decision Node)**: Gdy pod-węzeł dzieli się na kolejne pod-węzły.
4.  **Liść (Leaf/Terminal Node)**: Węzły, które nie są dalej dzielone (zawierają predykcję).

### Matematyka podziału

Aby zdecydować, jak najlepiej podzielić dane, używamy metryk "czystości" (purity).

#### 1. Entropia (Entropy)
Mierzy nieuporządkowanie w zbiorze danych.
$$ H(S) = - \sum_{i=1}^{c} p_i \log_2 p_i $$
Gdzie $p_i$ to prawdopodobieństwo wystąpienia klasy $i$.

#### 2. Zysk Informacyjny (Information Gain)
Mierzy spadek entropii po podziale zbioru $S$ na atrybut $A$.
$$ IG(S, A) = H(S) - \sum_{v \in Values(A)} \frac{|S_v|}{|S|} H(S_v) $$

#### 3. Indeks Giniego (Gini Impurity)
Alternatywa dla entropii (często szybsza w obliczeniach). Mierzy prawdopodobieństwo błędnej klasyfikacji losowego elementu.
$$ Gini = 1 - \sum_{i=1}^{c} (p_i)^2 $$

## 2. Implementacja (Implementation)

W tym katalogu znajdują się dwie implementacje:

1.  **`00_scratch.py`**: Własna implementacja algorytmu budowania drzewa (rekurencyjny podział).
2.  **`01_sklearn.py`**: Implementacja referencyjna używająca biblioteki `scikit-learn`.

### Wyniki

#### Scratch Implementation (Decision Boundary)
![Scratch Decision Tree](assets/scratch_tree_boundary.png)

#### Sklearn Implementation (Decision Boundary)
![Sklearn Decision Tree](assets/sklearn_tree_boundary.png)

## 3. Uruchomienie

```bash
python 00_scratch.py
python 01_sklearn.py
```
