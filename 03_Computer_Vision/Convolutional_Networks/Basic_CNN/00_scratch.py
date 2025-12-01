import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

# Ensure assets directory exists
script_dir = os.path.dirname(os.path.abspath(__file__))
assets_dir = os.path.join(script_dir, "assets")
os.makedirs(assets_dir, exist_ok=True)

def convolve2d(image, kernel):
    """
    Simple 2D convolution implementation (assuming stride=1, no padding for simplicity)
    """
    k_h, k_w = kernel.shape
    i_h, i_w = image.shape
    
    # Output dimensions
    o_h = i_h - k_h + 1
    o_w = i_w - k_w + 1
    
    output = np.zeros((o_h, o_w))
    
    for i in range(o_h):
        for j in range(o_w):
            # Extract region of interest
            region = image[i:i+k_h, j:j+k_w]
            # Element-wise multiplication and sum
            output[i, j] = np.sum(region * kernel)
            
    return output

def max_pooling(image, pool_size=2, stride=2):
    """
    Max Pooling implementation
    """
    i_h, i_w = image.shape
    o_h = (i_h - pool_size) // stride + 1
    o_w = (i_w - pool_size) // stride + 1
    
    output = np.zeros((o_h, o_w))
    
    for i in range(o_h):
        for j in range(o_w):
            h_start = i * stride
            h_end = h_start + pool_size
            w_start = j * stride
            w_end = w_start + pool_size
            
            region = image[h_start:h_end, w_start:w_end]
            output[i, j] = np.max(region)
            
    return output

# --- Visualization ---
if __name__ == "__main__":
    # 1. Create a synthetic image (Square)
    image = np.zeros((10, 10))
    image[2:8, 2:8] = 1.0
    
    # 2. Define Filters (Kernels)
    # Vertical Edge Detector
    sobel_v = np.array([[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]])
    
    # Horizontal Edge Detector
    sobel_h = np.array([[-1, -2, -1],
                        [ 0,  0,  0],
                        [ 1,  2,  1]])
    
    # 3. Apply Convolution
    conv_v = convolve2d(image, sobel_v)
    conv_h = convolve2d(image, sobel_h)
    
    # 4. Apply Pooling (to conv_v)
    pool_v = max_pooling(conv_v)
    
    # Plotting
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    
    # Row 1: Original & Filters
    axes[0, 0].imshow(image, cmap='gray')
    axes[0, 0].set_title("Original Image (10x10)")
    
    axes[0, 1].imshow(sobel_v, cmap='gray')
    axes[0, 1].set_title("Vertical Filter (3x3)")
    
    axes[0, 2].imshow(sobel_h, cmap='gray')
    axes[0, 2].set_title("Horizontal Filter (3x3)")
    
    # Row 2: Feature Maps & Pooling
    axes[1, 0].imshow(conv_v, cmap='gray')
    axes[1, 0].set_title("Vertical Conv Result")
    
    axes[1, 1].imshow(conv_h, cmap='gray')
    axes[1, 1].set_title("Horizontal Conv Result")
    
    axes[1, 2].imshow(pool_v, cmap='gray')
    axes[1, 2].set_title("Max Pooling (of Vert Conv)")
    
    for ax in axes.flat:
        ax.axis('off')
        
    output_path = os.path.join(assets_dir, "scratch_cnn_ops.png")
    plt.savefig(output_path)
    print(f"Saved visualization to {output_path}")
