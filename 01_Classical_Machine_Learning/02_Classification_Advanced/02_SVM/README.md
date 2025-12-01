# Support Vector Machines (SVM)

## 1. Teoria (Theory)

**Support Vector Machine (Maszyna Wektorów Nośnych)** to potężny algorytm klasyfikacji, który szuka optymalnej hiperpłaszczyzny (hyperplane) oddzielającej klasy z maksymalnym marginesem.

### Kluczowe koncepcje:

1.  **Hiperpłaszczyzna (Hyperplane)**: Granica decyzyjna oddzielająca klasy. W 2D to linia, w 3D płaszczyzna.
    $$ w \cdot x - b = 0 $$
2.  **Margines (Margin)**: Odległość między hiperpłaszczyzną a najbliższymi punktami danych z każdej klasy. SVM dąży do maksymalizacji tego marginesu.
3.  **Wektory Nośne (Support Vectors)**: Punkty danych leżące najbliżej hiperpłaszczyzny. To one "podtrzymują" lub definiują margines.
4.  **Kernel Trick**: Metoda mapowania danych do wyższego wymiaru, aby umożliwić separację liniową danych, które w oryginalnym wymiarze nie są liniowo separowalne (np. kernel RBF).

### Matematyka (Linear SVM - Soft Margin)

Chcemy zminimalizować normę wektora wag $||w||$ (co maksymalizuje margines) przy jednoczesnym karaniu za błędy klasyfikacji (punkty po złej stronie marginesu).

Funkcja kosztu (Hinge Loss + Regularyzacja):

$$ J(w, b) = \frac{1}{2} ||w||^2 + C \sum_{i=1}^{n} \max(0, 1 - y_i(w \cdot x_i - b)) $$

Gdzie:
*   $y_i \in \{-1, 1\}$ to etykiety klas.
*   $C$ to parametr regularyzacji (balans między szerokością marginesu a błędami klasyfikacji).

### Gradient Descent Update Rule

Dla każdego przykładu $i$:
Jeśli $y_i(w \cdot x_i - b) \ge 1$ (poprawnie sklasyfikowany z marginesem):
$$ w = w - \alpha (2\lambda w) $$
Jeśli $y_i(w \cdot x_i - b) < 1$ (błąd lub wewnątrz marginesu):
$$ w = w - \alpha (2\lambda w - y_i x_i) $$
$$ b = b - \alpha (y_i) $$

*(Uwaga: W implementacji scratch użyjemy uproszczonej wersji z $\lambda = 1/C$)*

## 2. Implementacja (Implementation)

1.  **`00_scratch.py`**: Implementacja Liniowego SVM przy użyciu metody spadku gradientu (Gradient Descent) minimalizującego funkcję Hinge Loss.
2.  **`01_sklearn.py`**: Porównanie SVM z jądrem liniowym (Linear) oraz nieliniowym (RBF - Radial Basis Function) przy użyciu `scikit-learn`.

### Wyniki

#### Scratch Implementation (Linear SVM)
![Scratch SVM](assets/scratch_svm_boundary.png)

#### Sklearn Implementation (Linear vs RBF)
![Sklearn SVM](assets/sklearn_svm_comparison.png)

## 3. Uruchomienie

```bash
python 00_scratch.py
python 01_sklearn.py
```
