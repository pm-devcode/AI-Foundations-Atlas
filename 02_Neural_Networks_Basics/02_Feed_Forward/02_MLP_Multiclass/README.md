# Multi-Layer Perceptron (MLP) - Multiclass Classification

## 1. Teoria (Theory)

W klasyfikacji wieloklasowej (gdy mamy $C > 2$ klas), architektura sieci ulega zmianie głównie w warstwie wyjściowej.

### Kluczowe zmiany:

1.  **Warstwa Wyjściowa**: Liczba neuronów równa się liczbie klas $C$.
2.  **Funkcja Aktywacji (Softmax)**: Zamiast Sigmoid (który zwraca wartość 0-1 niezależnie dla każdego neuronu), używamy Softmax, który zamienia wektor surowych wyników (logits) na rozkład prawdopodobieństwa.
    $$ \text{Softmax}(z_i) = \frac{e^{z_i}}{\sum_{j=1}^{C} e^{z_j}} $$
    Suma wszystkich wyjść wynosi 1.

3.  **Funkcja Kosztu (Categorical Cross-Entropy)**:
    $$ L = - \sum_{i=1}^{C} y_i \log(\hat{y}_i) $$
    Gdzie $y$ to wektor one-hot (np. `[0, 1, 0]`), a $\hat{y}$ to predykcje sieci.

### One-Hot Encoding

Etykiety klas są zamieniane na wektory zer i jedynek.
*   Klasa 0 -> `[1, 0, 0]`
*   Klasa 1 -> `[0, 1, 0]`
*   Klasa 2 -> `[0, 0, 1]`

## 2. Implementacja (Implementation)

Użyjemy syntetycznego zbioru danych z 3 klasami, aby łatwo zwizualizować granice decyzyjne w 2D.

1.  **`00_scratch.py`**: Implementacja MLP z obsługą macierzy (batch processing), funkcją Softmax i pochodną Cross-Entropy.
2.  **`01_pytorch.py`**: Implementacja w PyTorch używająca `nn.CrossEntropyLoss` (która łączy `LogSoftmax` i `NLLLoss` dla stabilności numerycznej).

### Wyniki

#### Scratch Implementation (Multiclass Boundaries)
![Scratch MLP Multiclass](assets/scratch_mlp_multiclass.png)

#### PyTorch Implementation (Multiclass Boundaries)
![PyTorch MLP Multiclass](assets/pytorch_mlp_multiclass.png)

## 3. Uruchomienie

```bash
python 00_scratch.py
python 01_pytorch.py
```
