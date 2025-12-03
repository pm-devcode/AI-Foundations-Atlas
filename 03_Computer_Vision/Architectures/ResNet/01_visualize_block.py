import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

# Ensure assets directory exists
script_dir = os.path.dirname(os.path.abspath(__file__))
assets_dir = os.path.join(script_dir, "assets")
os.makedirs(assets_dir, exist_ok=True)

def draw_residual_block():
    fig, ax = plt.subplots(figsize=(6, 8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 12)
    ax.axis('off')

    # Style
    box_props = dict(boxstyle='round,pad=0.5', facecolor='lightblue', edgecolor='black')
    op_props = dict(boxstyle='circle,pad=0.3', facecolor='lightgreen', edgecolor='black')
    
    # Input x
    ax.text(5, 11, 'x', ha='center', va='center', fontsize=14, fontweight='bold')
    ax.arrow(5, 10.5, 0, -1, head_width=0.2, head_length=0.3, fc='k', ec='k')

    # Split point
    ax.plot(5, 9.5, 'ko', markersize=5)
    
    # Main Path (Weight Layer)
    ax.text(5, 8, 'Weight Layer\n(Conv + BN + ReLU)', ha='center', va='center', bbox=box_props)
    ax.arrow(5, 9.5, 0, -0.8, head_width=0.2, head_length=0.3, fc='k', ec='k')
    ax.arrow(5, 7.2, 0, -0.8, head_width=0.2, head_length=0.3, fc='k', ec='k')

    # Second Weight Layer
    ax.text(5, 5.5, 'Weight Layer\n(Conv + BN)', ha='center', va='center', bbox=box_props)
    ax.arrow(5, 4.7, 0, -0.8, head_width=0.2, head_length=0.3, fc='k', ec='k')

    # Addition
    ax.text(5, 3, '+', ha='center', va='center', bbox=op_props, fontsize=14)
    
    # Skip Connection
    # Draw a curved line from split point to addition
    # Using Arc or just plotting points
    # Path: (5, 9.5) -> (8, 9.5) -> (8, 3) -> (5.5, 3)
    ax.plot([5, 8], [9.5, 9.5], 'k-', lw=1.5) # Right
    ax.plot([8, 8], [9.5, 3], 'k-', lw=1.5)   # Down
    ax.arrow(8, 3, -2.4, 0, head_width=0.2, head_length=0.3, fc='k', ec='k') # Left to +
    
    ax.text(8.5, 6, 'Identity\nMapping\n(Skip Connection)', ha='center', va='center', fontsize=10, style='italic')

    # Output
    ax.text(5, 2, 'ReLU', ha='center', va='center', bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='black'))
    ax.arrow(5, 2.6, 0, -0.3, head_width=0.2, head_length=0.3, fc='k', ec='k')
    ax.arrow(5, 1.7, 0, -0.7, head_width=0.2, head_length=0.3, fc='k', ec='k')
    
    ax.text(5, 0.5, 'F(x) + x', ha='center', va='center', fontsize=14, fontweight='bold')

    plt.title("Residual Block Architecture", fontsize=16)
    
    output_path = os.path.join(assets_dir, "residual_block.png")
    plt.savefig(output_path)
    print(f"Saved visualization to {output_path}")

if __name__ == "__main__":
    draw_residual_block()
