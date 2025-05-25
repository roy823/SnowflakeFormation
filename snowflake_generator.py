import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from enhanced_ca import EnhancedReiterCA

def generate_snowflake(pattern_name, grid_size=151, save_dir="snowflakes", animate=False):
    """
    Generate specific snowflake crystal pattern
    
    Parameters:
        pattern_name: Preset pattern name
        grid_size: Grid size
        save_dir: Save directory
        animate: Whether to generate animation
    """
    # Create save directory
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    # Preset pattern parameters
    patterns = {
        "dendrite": {
            "alpha": 0.7,
            "beta": 0.4,
            "gamma": 0.0008,
            "seed_type": "single",
            "color_scheme": "default",
            "contrast": 0.5,
            "steps": 800
        },
        "fernlike": {
            "alpha": 1.0,
            "beta": 0.7, 
            "gamma": 0.0012,
            "seed_type": "star",
            "color_scheme": "winter",
            "contrast": 0.7,
            "steps": 700
        },
        "plate": {
            "alpha": 0.5,
            "beta": 0.6,
            "gamma": 0.002,
            "seed_type": "ring",
            "color_scheme": "default",
            "contrast": 0.3,
            "steps": 600
        },
        "stellar": {
            "alpha": 0.8,
            "beta": 0.5,
            "gamma": 0.001,
            "seed_type": "star",
            "color_scheme": "rainbow",
            "contrast": 0.4,
            "steps": 850
        },
        "sectored": {
            "alpha": 0.9,
            "beta": 0.3,
            "gamma": 0.004,
            "seed_type": "triple",
            "color_scheme": "hot",
            "contrast": 0.6,
            "steps": 750
        }
    }
    
    # Check parameter validity
    if pattern_name not in patterns:
        raise ValueError(f"Unknown pattern name: {pattern_name}. Available options: {list(patterns.keys())}")
    
    # Get parameters
    params = patterns[pattern_name]
    
    # Create CA model
    ca = EnhancedReiterCA(
        grid_size=grid_size, 
        alpha=params["alpha"], 
        beta=params["beta"], 
        gamma=params["gamma"],
        seed_type=params["seed_type"],
        color_scheme=params["color_scheme"]
    )
    
    # Generate animation or static image
    if animate:
        fig, ax = plt.subplots(figsize=(10, 10))
        
        def update(frame):
            ax.clear()
            ca.draw(ax, contrast_param=params["contrast"])
            ax.set_title(f"Snow Crystal Pattern: {pattern_name} (Step {frame})")
            ca.update()
            return ax,
            
        anim = FuncAnimation(fig, update, frames=range(params["steps"]), 
                             interval=100, blit=True)
        
        # Save animation
        anim_path = os.path.join(save_dir, f"{pattern_name}_animation.gif")
        anim.save(anim_path, writer='pillow', fps=10, dpi=100)
        return anim_path
    
    else:
        # Run simulation
        for _ in range(params["steps"]):
            ca.update()
            if ca.edge_touched():
                break
                
        # Generate final image
        fig, ax = plt.subplots(figsize=(10, 10))
        ca.draw(ax, contrast_param=params["contrast"])
        ax.set_title(f"Snow Crystal Pattern: {pattern_name}")
        
        # Save image
        image_path = os.path.join(save_dir, f"{pattern_name}.png")
        plt.savefig(image_path, dpi=300, bbox_inches='tight')
        plt.close()
        return image_path

# Generate all types of snowflakes
def generate_all_snowflakes(grid_size=151, save_dir="snowflakes"):
    """Generate all preset snowflake patterns"""
    results = {}
    for pattern in ["dendrite", "fernlike", "plate", "stellar", "sectored"]:
        print(f"Generating {pattern} snow crystal pattern...")
        path = generate_snowflake(pattern, grid_size, save_dir)
        results[pattern] = path
    return results