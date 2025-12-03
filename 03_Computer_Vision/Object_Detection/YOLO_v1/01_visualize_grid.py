import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os

# Ensure assets directory exists
script_dir = os.path.dirname(os.path.abspath(__file__))
assets_dir = os.path.join(script_dir, "assets")
os.makedirs(assets_dir, exist_ok=True)

def draw_yolo_concept():
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Create a blank image (white background)
    img_size = 448
    ax.set_xlim(0, img_size)
    ax.set_ylim(img_size, 0) # Flip y-axis to match image coords
    ax.set_aspect('equal')
    
    # Draw Grid (7x7)
    S = 7
    step = img_size / S
    
    for i in range(S + 1):
        # Horizontal lines
        ax.plot([0, img_size], [i * step, i * step], 'k-', lw=0.5, alpha=0.5)
        # Vertical lines
        ax.plot([i * step, i * step], [0, img_size], 'k-', lw=0.5, alpha=0.5)
        
    # Draw an "Object" (e.g., a Dog represented by a circle/ellipse)
    # Center it in grid cell (3, 3) -> indices (2, 2) if 0-indexed?
    # Let's put it in cell row=3, col=4 (0-indexed)
    # Cell coords: x=[4*step, 5*step], y=[3*step, 4*step]
    
    obj_x_center = 4.5 * step
    obj_y_center = 3.5 * step
    obj_width = 1.5 * step
    obj_height = 1.2 * step
    
    # Draw object
    ellipse = patches.Ellipse((obj_x_center, obj_y_center), obj_width, obj_height, 
                              facecolor='orange', alpha=0.6, edgecolor='none')
    ax.add_patch(ellipse)
    ax.text(obj_x_center, obj_y_center, "Object", ha='center', va='center', fontweight='bold')
    
    # Highlight the responsible grid cell
    cell_x = 4 * step
    cell_y = 3 * step
    rect = patches.Rectangle((cell_x, cell_y), step, step, linewidth=2, edgecolor='red', facecolor='none')
    ax.add_patch(rect)
    ax.text(cell_x + 5, cell_y + 15, "Responsible\nCell", color='red', fontsize=10, fontweight='bold')
    
    # Draw Bounding Box Prediction
    # Bounding box is relative to the cell center usually, but visualized globally
    bbox = patches.Rectangle((obj_x_center - obj_width/2, obj_y_center - obj_height/2), 
                             obj_width, obj_height, linewidth=2, edgecolor='blue', facecolor='none', linestyle='--')
    ax.add_patch(bbox)
    ax.text(obj_x_center - obj_width/2, obj_y_center - obj_height/2 - 5, "Predicted BBox (x, y, w, h)", color='blue', fontsize=10)
    
    # Center point
    ax.plot(obj_x_center, obj_y_center, 'rx', markersize=8)
    
    plt.title(f"YOLO Concept: {S}x{S} Grid", fontsize=16)
    plt.axis('off')
    
    output_path = os.path.join(assets_dir, "yolo_grid.png")
    plt.savefig(output_path)
    print(f"Saved visualization to {output_path}")

if __name__ == "__main__":
    draw_yolo_concept()
