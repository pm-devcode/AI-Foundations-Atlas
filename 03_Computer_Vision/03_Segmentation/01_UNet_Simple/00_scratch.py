import numpy as np
import os

# Ensure assets directory exists (even if not used for plotting here, good practice)
script_dir = os.path.dirname(os.path.abspath(__file__))
assets_dir = os.path.join(script_dir, "assets")
os.makedirs(assets_dir, exist_ok=True)

def print_tensor_info(name, tensor):
    print(f"[{name}] Shape: {tensor.shape}")

def conv_block_sim(x, out_channels):
    """Simulates shape change of a Conv Block (Conv->ReLU->Conv->ReLU)"""
    # Assuming padding='same', shape doesn't change spatially
    b, h, w, c = x.shape
    return np.zeros((b, h, w, out_channels))

def max_pool_sim(x):
    """Simulates shape change of Max Pooling (halves spatial dims)"""
    b, h, w, c = x.shape
    return np.zeros((b, h // 2, w // 2, c))

def upsample_sim(x):
    """Simulates shape change of UpSampling (doubles spatial dims)"""
    b, h, w, c = x.shape
    return np.zeros((b, h * 2, w * 2, c))

def concat_sim(x1, x2):
    """Simulates Skip Connection Concatenation"""
    # x1 is from decoder (upsampled), x2 is from encoder (skip connection)
    # Concatenate along channel axis (last axis in numpy usually, but let's assume NHWC for sim)
    return np.concatenate([x1, x2], axis=-1)

# --- U-Net Forward Pass Simulation ---
if __name__ == "__main__":
    print("--- U-Net Architecture Shape Simulation (NHWC format) ---\n")
    
    # Input Image: Batch=1, Height=128, Width=128, Channels=1
    input_img = np.zeros((1, 128, 128, 1))
    print_tensor_info("Input Image", input_img)
    
    # --- Encoder (Contracting Path) ---
    print("\n--- Encoder ---")
    # Block 1
    c1 = conv_block_sim(input_img, 64)
    p1 = max_pool_sim(c1)
    print_tensor_info("Encoder Block 1 (Conv)", c1)
    print_tensor_info("Encoder Block 1 (Pool)", p1)
    
    # Block 2
    c2 = conv_block_sim(p1, 128)
    p2 = max_pool_sim(c2)
    print_tensor_info("Encoder Block 2 (Conv)", c2)
    print_tensor_info("Encoder Block 2 (Pool)", p2)
    
    # Block 3
    c3 = conv_block_sim(p2, 256)
    p3 = max_pool_sim(c3)
    print_tensor_info("Encoder Block 3 (Conv)", c3)
    print_tensor_info("Encoder Block 3 (Pool)", p3)
    
    # --- Bottleneck ---
    print("\n--- Bottleneck ---")
    bn = conv_block_sim(p3, 512)
    print_tensor_info("Bottleneck", bn)
    
    # --- Decoder (Expanding Path) ---
    print("\n--- Decoder ---")
    
    # Up-Block 3
    u3 = upsample_sim(bn)
    print_tensor_info("Upsample 3", u3)
    # Skip Connection from c3
    cat3 = concat_sim(u3, c3)
    print(f"Concatenation 3: {u3.shape} + {c3.shape} -> {cat3.shape}")
    d3 = conv_block_sim(cat3, 256)
    print_tensor_info("Decoder Block 3", d3)
    
    # Up-Block 2
    u2 = upsample_sim(d3)
    print_tensor_info("Upsample 2", u2)
    # Skip Connection from c2
    cat2 = concat_sim(u2, c2)
    print(f"Concatenation 2: {u2.shape} + {c2.shape} -> {cat2.shape}")
    d2 = conv_block_sim(cat2, 128)
    print_tensor_info("Decoder Block 2", d2)
    
    # Up-Block 1
    u1 = upsample_sim(d2)
    print_tensor_info("Upsample 1", u1)
    # Skip Connection from c1
    cat1 = concat_sim(u1, c1)
    print(f"Concatenation 1: {u1.shape} + {c1.shape} -> {cat1.shape}")
    d1 = conv_block_sim(cat1, 64)
    print_tensor_info("Decoder Block 1", d1)
    
    # --- Output Layer ---
    print("\n--- Output ---")
    # 1x1 Convolution to get desired number of classes (1 for binary mask)
    output = np.zeros((1, 128, 128, 1)) 
    print_tensor_info("Output Mask", output)
    
    print("\nSimulation Complete. Dimensions match U-Net logic.")
