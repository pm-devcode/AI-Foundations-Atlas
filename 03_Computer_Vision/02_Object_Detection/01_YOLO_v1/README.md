# YOLO v1 (You Only Look Once)

## 1. Concept
YOLO is a unified architecture for object detection. Unlike prior systems (R-CNN) that repurpose classifiers to perform detection, YOLO frames object detection as a **regression problem** to spatially separated bounding boxes and associated class probabilities.

### Key Idea: The Grid
*   The input image is divided into an $S \times S$ grid.
*   If the center of an object falls into a grid cell, that grid cell is responsible for detecting that object.
*   Each grid cell predicts $B$ bounding boxes and confidence scores for those boxes.
*   Each grid cell also predicts $C$ conditional class probabilities.

## 2. Architecture
The network is inspired by GoogLeNet.
*   **Input**: $448 \times 448 \times 3$ image.
*   **Output**: $S \times S \times (B \times 5 + C)$ tensor.
    *   For PASCAL VOC: $S=7, B=2, C=20$.
    *   Output size: $7 \times 7 \times 30$.

### Output Tensor Structure (per cell)
Each cell outputs a vector of length 30:
*   **Indices 0-19**: Class probabilities (20 classes).
*   **Indices 20-24**: Bounding Box 1 ($P_c, x, y, w, h$).
*   **Indices 25-29**: Bounding Box 2 ($P_c, x, y, w, h$).

## 3. Loss Function
The loss function is a sum of squared errors, but weighted to handle the class imbalance (many background cells).

$$ \lambda_{coord} \sum_{i=0}^{S^2} \sum_{j=0}^{B} \mathbb{1}_{ij}^{obj} [(x_i - \hat{x}_i)^2 + (y_i - \hat{y}_i)^2] $$
$$ + \lambda_{coord} \sum_{i=0}^{S^2} \sum_{j=0}^{B} \mathbb{1}_{ij}^{obj} [(\sqrt{w_i} - \sqrt{\hat{w}_i})^2 + (\sqrt{h_i} - \sqrt{\hat{h}_i})^2] $$
$$ + \sum_{i=0}^{S^2} \sum_{j=0}^{B} \mathbb{1}_{ij}^{obj} (C_i - \hat{C}_i)^2 $$
$$ + \lambda_{noobj} \sum_{i=0}^{S^2} \sum_{j=0}^{B} \mathbb{1}_{ij}^{noobj} (C_i - \hat{C}_i)^2 $$
$$ + \sum_{i=0}^{S^2} \mathbb{1}_{i}^{obj} \sum_{c \in classes} (p_i(c) - \hat{p}_i(c))^2 $$

## 4. Implementation Details
*   **`00_model.py`**: Contains the full PyTorch implementation of the YOLO v1 architecture and the custom Loss function.
*   **Note**: Training this model requires a large dataset (PASCAL VOC) and significant compute. This implementation serves as an architectural reference.

## 5. How to Run
```bash
# Verify model architecture
python 00_model.py
```
