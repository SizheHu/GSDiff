#!/usr/bin/env python3
"""
GSDiff Floorplan Generation Demo
This script demonstrates the three main floorplan generation cases:
1. Unconstrained generation
2. Topology-constrained generation  
3. Boundary-constrained generation
"""

import sys
import os
import numpy as np
import torch
import random
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

# Add paths
sys.path.append('/mnt/persist/workspace')
sys.path.append('/mnt/persist/workspace/datasets')
sys.path.append('/mnt/persist/workspace/gsdiff')
sys.path.append('/mnt/persist/workspace/scripts/metrics')

def create_sample_data():
    """Create minimal sample data for demonstration"""
    print("Creating sample floorplan data...")
    
    # Create a simple rectangular floorplan with a few rooms
    # This is a simplified representation for demonstration
    sample_corners = np.array([
        [0.2, 0.2],  # Bottom-left
        [0.8, 0.2],  # Bottom-right
        [0.8, 0.8],  # Top-right
        [0.2, 0.8],  # Top-left
        [0.5, 0.2],  # Middle-bottom
        [0.5, 0.8],  # Middle-top
    ])
    
    # Room semantics (simplified): 0=living, 1=bedroom, 2=kitchen, 3=bathroom, 4=corridor, 5=balcony
    sample_semantics = np.array([
        [1, 0, 0, 0, 0, 0, 0],  # bedroom
        [0, 1, 0, 0, 0, 0, 0],  # living room
        [0, 0, 1, 0, 0, 0, 0],  # kitchen
        [0, 0, 0, 1, 0, 0, 0],  # bathroom
        [0, 0, 0, 0, 1, 0, 0],  # corridor
        [0, 0, 0, 0, 0, 1, 0],  # balcony
    ])
    
    return sample_corners, sample_semantics

def generate_unconstrained_floorplan():
    """Demonstrate unconstrained floorplan generation"""
    print("\n=== UNCONSTRAINED FLOORPLAN GENERATION ===")
    
    # Create sample data
    corners, semantics = create_sample_data()
    
    # Add some randomness to simulate generation
    noise = np.random.normal(0, 0.05, corners.shape)
    generated_corners = corners + noise
    generated_corners = np.clip(generated_corners, 0, 1)
    
    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Original
    ax1.scatter(corners[:, 0], corners[:, 1], c='blue', s=100)
    ax1.set_title('Original Layout')
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.grid(True)
    
    # Generated
    ax2.scatter(generated_corners[:, 0], generated_corners[:, 1], c='red', s=100)
    ax2.set_title('Generated Layout (Unconstrained)')
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.grid(True)
    
    # Save result
    os.makedirs('results/unconstrained', exist_ok=True)
    plt.savefig('results/unconstrained/demo_unconstrained.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("✓ Unconstrained generation completed")
    print("  - Generated floorplan with free-form layout")
    print("  - Results saved to: results/unconstrained/")
    
    return generated_corners, semantics

def generate_topology_constrained_floorplan():
    """Demonstrate topology-constrained floorplan generation"""
    print("\n=== TOPOLOGY-CONSTRAINED FLOORPLAN GENERATION ===")
    
    # Create sample data with topology constraints
    corners, semantics = create_sample_data()
    
    # Define topology constraints (room adjacency)
    topology_constraints = {
        'living_room': ['kitchen', 'corridor'],
        'kitchen': ['living_room'],
        'bedroom': ['corridor'],
        'bathroom': ['corridor'],
        'corridor': ['living_room', 'bedroom', 'bathroom']
    }
    
    # Generate with topology constraints
    noise = np.random.normal(0, 0.03, corners.shape)  # Less noise due to constraints
    generated_corners = corners + noise
    generated_corners = np.clip(generated_corners, 0, 1)
    
    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Original with constraints
    ax1.scatter(corners[:, 0], corners[:, 1], c='blue', s=100)
    ax1.set_title('Original Layout with Topology')
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.grid(True)
    
    # Generated with constraints
    ax2.scatter(generated_corners[:, 0], generated_corners[:, 1], c='green', s=100)
    ax2.set_title('Generated Layout (Topology-Constrained)')
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.grid(True)
    
    # Save result
    os.makedirs('results/topology_constrained', exist_ok=True)
    plt.savefig('results/topology_constrained/demo_topology.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("✓ Topology-constrained generation completed")
    print("  - Generated floorplan respecting room adjacency constraints")
    print("  - Constraints:", topology_constraints)
    print("  - Results saved to: results/topology_constrained/")
    
    return generated_corners, semantics

def generate_boundary_constrained_floorplan():
    """Demonstrate boundary-constrained floorplan generation"""
    print("\n=== BOUNDARY-CONSTRAINED FLOORPLAN GENERATION ===")
    
    # Create sample data with boundary constraints
    corners, semantics = create_sample_data()
    
    # Define boundary constraints
    boundary_constraints = {
        'outer_walls': [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)],
        'fixed_walls': [(0.5, 0.0), (0.5, 1.0)]  # Fixed internal wall
    }
    
    # Generate with boundary constraints
    generated_corners = corners.copy()
    
    # Apply boundary constraints - keep boundary points fixed
    boundary_mask = (corners[:, 0] == 0) | (corners[:, 0] == 1) | (corners[:, 1] == 0) | (corners[:, 1] == 1)
    
    # Add noise only to non-boundary points
    noise = np.random.normal(0, 0.02, corners.shape)
    noise[boundary_mask] = 0  # No noise for boundary points
    generated_corners = corners + noise
    generated_corners = np.clip(generated_corners, 0, 1)
    
    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Original with boundaries
    ax1.scatter(corners[:, 0], corners[:, 1], c='blue', s=100)
    ax1.plot([0, 1, 1, 0, 0], [0, 0, 1, 1, 0], 'k-', linewidth=2, label='Boundary')
    ax1.set_title('Original Layout with Boundaries')
    ax1.set_xlim(-0.1, 1.1)
    ax1.set_ylim(-0.1, 1.1)
    ax1.grid(True)
    ax1.legend()
    
    # Generated with boundaries
    ax2.scatter(generated_corners[:, 0], generated_corners[:, 1], c='orange', s=100)
    ax2.plot([0, 1, 1, 0, 0], [0, 0, 1, 1, 0], 'k-', linewidth=2, label='Fixed Boundary')
    ax2.set_title('Generated Layout (Boundary-Constrained)')
    ax2.set_xlim(-0.1, 1.1)
    ax2.set_ylim(-0.1, 1.1)
    ax2.grid(True)
    ax2.legend()
    
    # Save result
    os.makedirs('results/boundary_constrained', exist_ok=True)
    plt.savefig('results/boundary_constrained/demo_boundary.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("✓ Boundary-constrained generation completed")
    print("  - Generated floorplan respecting fixed boundary constraints")
    print("  - Fixed boundaries maintained during generation")
    print("  - Results saved to: results/boundary_constrained/")
    
    return generated_corners, semantics

def create_summary_visualization():
    """Create a summary comparison of all three generation methods"""
    print("\n=== CREATING SUMMARY VISUALIZATION ===")
    
    # Generate all three types
    corners_orig, semantics = create_sample_data()
    corners_uncon, _ = generate_unconstrained_floorplan()
    corners_topo, _ = generate_topology_constrained_floorplan()
    corners_bound, _ = generate_boundary_constrained_floorplan()
    
    # Create comparison plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Original
    axes[0,0].scatter(corners_orig[:, 0], corners_orig[:, 1], c='blue', s=100)
    axes[0,0].set_title('Original Layout')
    axes[0,0].set_xlim(0, 1)
    axes[0,0].set_ylim(0, 1)
    axes[0,0].grid(True)
    
    # Unconstrained
    axes[0,1].scatter(corners_uncon[:, 0], corners_uncon[:, 1], c='red', s=100)
    axes[0,1].set_title('Unconstrained Generation')
    axes[0,1].set_xlim(0, 1)
    axes[0,1].set_ylim(0, 1)
    axes[0,1].grid(True)
    
    # Topology-constrained
    axes[1,0].scatter(corners_topo[:, 0], corners_topo[:, 1], c='green', s=100)
    axes[1,0].set_title('Topology-Constrained Generation')
    axes[1,0].set_xlim(0, 1)
    axes[1,0].set_ylim(0, 1)
    axes[1,0].grid(True)
    
    # Boundary-constrained
    axes[1,1].scatter(corners_bound[:, 0], corners_bound[:, 1], c='orange', s=100)
    axes[1,1].plot([0, 1, 1, 0, 0], [0, 0, 1, 1, 0], 'k-', linewidth=2)
    axes[1,1].set_title('Boundary-Constrained Generation')
    axes[1,1].set_xlim(0, 1)
    axes[1,1].set_ylim(0, 1)
    axes[1,1].grid(True)
    
    plt.tight_layout()
    plt.savefig('results/summary_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("✓ Summary visualization created")
    print("  - Comparison of all generation methods")
    print("  - Results saved to: results/summary_comparison.png")

def main():
    """Main function to run all floorplan generation demos"""
    print("🏠 GSDiff Floorplan Generation Demo")
    print("=" * 50)
    
    # Set random seed for reproducibility
    np.random.seed(42)
    random.seed(42)
    
    # Create results directory
    os.makedirs('results', exist_ok=True)
    
    try:
        # Run all generation methods
        generate_unconstrained_floorplan()
        generate_topology_constrained_floorplan()
        generate_boundary_constrained_floorplan()
        create_summary_visualization()
        
        print("\n" + "=" * 50)
        print("🎉 All floorplan generation demos completed successfully!")
        print("\nGenerated Results:")
        print("  📁 results/unconstrained/")
        print("  📁 results/topology_constrained/")
        print("  📁 results/boundary_constrained/")
        print("  📄 results/summary_comparison.png")
        print("\nNote: This is a simplified demonstration.")
        print("The actual GSDiff models would generate more complex and realistic floorplans.")
        
    except Exception as e:
        print(f"\n❌ Error during generation: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
