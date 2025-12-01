# U-Net (Semantic Segmentation)

## 1. Executive Summary
U-Net is a specialized Convolutional Neural Network architecture designed for **Semantic Segmentation**—the task of assigning a class label to every single pixel in an image (e.g., "this pixel is a car", "this pixel is the road"). It is famous for its "U" shape, consisting of a contracting path (encoder) and an expanding path (decoder).

## 2. Historical Context
U-Net was introduced by **Olaf Ronneberger, Philipp Fischer, and Thomas Brox** in **2015** at the University of Freiburg. It was originally designed for **biomedical image segmentation** (e.g., detecting cells in microscopy images) where labeled data is scarce. Its ability to work with few training images and produce precise segmentations made it a standard in the field and beyond.

## 3. Real-World Analogy
Think of an **Art Restoration Expert**.
*   **Encoder (Analysis)**: The expert studies the painting, understanding the high-level composition ("This is a portrait of a woman"). They might squint or step back to see the big picture, losing some fine details.
*   **Decoder (Restoration)**: The expert starts repainting the damaged areas. They need to fill in the details.
*   **Skip Connections**: To do this accurately, the expert constantly refers back to the original high-resolution sketches or photos (features from the encoder) to ensure the edges of the eyes or the texture of the hair are placed exactly right. Without these "peeks" at the original details, the restoration would be blurry.

## 4. Key Concepts

1.  **Contracting Path (Encoder)**: Typical CNN (Conv -> ReLU -> MaxPool). Reduces spatial dimensions, increases feature channels. Captures "What".
2.  **Expanding Path (Decoder)**: Uses Up-Convolution (Transposed Conv) to increase spatial dimensions. Captures "Where".
3.  **Skip Connections**: Concatenating feature maps from the encoder to the decoder. This recovers spatial information lost during pooling.

## 5. Implementation Details

1.  **`00_scratch.py`**: Simulation of the "Forward Pass" in pure NumPy.
    *   Demonstrates how tensor shapes change through the network.
    *   Shows the mechanism of Skip Connections (concatenation).
2.  **`01_pytorch.py`**: Full training of a simplified U-Net.
    *   **Data**: Synthetic images with random circles.
    *   **Task**: Generate a binary mask (Circle vs Background).
    *   **Loss**: `BCEWithLogitsLoss` (Binary Cross Entropy per pixel).

## 6. Results

### PyTorch Segmentation Results
![PyTorch U-Net Segmentation](assets/pytorch_unet_segmentation.png)

*(Input Image | Ground Truth Mask | Predicted Mask)*

## 7. How to Run

```bash
python 00_scratch.py
python 01_pytorch.py
```
