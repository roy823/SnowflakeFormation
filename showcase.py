import matplotlib.pyplot as plt
import numpy as np
import argparse
from enhanced_ca import EnhancedReiterCA
from snowflake_generator import generate_snowflake, generate_all_snowflakes

def custom_snowflake(grid_size=151, alpha=0.7, beta=0.5, gamma=0.001, 
                    seed_type="single", color_scheme="default", steps=800):
    """Create a custom snowflake with specified parameters"""
    ca = EnhancedReiterCA(
        grid_size=grid_size,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        seed_type=seed_type,
        color_scheme=color_scheme
    )
    
    # Run simulation
    for i in range(steps):
        ca.update()
        if i % 100 == 0:
            print(f"Step {i}/{steps}")
        if ca.edge_touched():
            print(f"Edge touched at step {i}")
            break
    
    fig, ax = plt.subplots(figsize=(10, 10))
    ca.draw(ax)
    plt.title(f"Custom Snowflake (α={alpha}, β={beta}, γ={gamma}, seed={seed_type}, color={color_scheme})")
    plt.tight_layout()
    plt.savefig("custom_snowflake.png", dpi=300)
    plt.show()

def showcase_all():
    """Show all preset snowflake patterns"""
    generate_all_snowflakes()
    
    # Create a showcase figure
    fig, axs = plt.subplots(2, 3, figsize=(15, 10))
    axs = axs.flatten()
    
    patterns = ["dendrite", "fernlike", "plate", "stellar", "sectored"]
    for i, pattern in enumerate(patterns):
        img = plt.imread(f"snowflakes/{pattern}.png")
        axs[i].imshow(img)
        axs[i].set_title(pattern)
        axs[i].axis('off')
    
    # Hide extra subplots
    if len(axs) > len(patterns):
        axs[-1].axis('off')
    
    plt.tight_layout()
    plt.savefig("snowflakes/showcase.png", dpi=300)
    plt.show()

def create_animation(pattern="dendrite"):
    """Create animation for specific pattern"""
    generate_snowflake(pattern, animate=True)
    print(f"Animation saved to snowflakes/{pattern}_animation.gif")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Snowflake Generator Showcase")
    parser.add_argument("--mode", choices=["showcase", "custom", "animate", "all"], 
                        default="showcase", help="Run mode")
    parser.add_argument("--pattern", default="dendrite", help="Preset pattern name")
    parser.add_argument("--alpha", type=float, default=0.7, help="Diffusion parameter")
    parser.add_argument("--beta", type=float, default=0.5, help="Background vapor parameter")
    parser.add_argument("--gamma", type=float, default=0.001, help="Crystallization parameter")
    parser.add_argument("--seed", default="single", 
                        choices=["single", "double", "triple", "star", "ring"], 
                        help="Seed type")
    parser.add_argument("--color", default="default", 
                        choices=["default", "winter", "hot", "rainbow"], 
                        help="Color scheme")
    parser.add_argument("--grid", type=int, default=151, help="Grid size")
    args = parser.parse_args()
    
    if args.mode == "showcase":
        showcase_all()
    elif args.mode == "custom":
        custom_snowflake(args.grid, args.alpha, args.beta, args.gamma, args.seed, args.color)
    elif args.mode == "animate":
        create_animation(args.pattern)
    elif args.mode == "all":
        for pattern in ["dendrite", "fernlike", "plate", "stellar", "sectored"]:
            create_animation(pattern)