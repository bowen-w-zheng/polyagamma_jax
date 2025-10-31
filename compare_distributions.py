"""
Compare distributions obtained by different sampling methods.

This script compares the three JAX sampling methods (Devroye, Saddle, Normal)
against the C-backend implementation across different z values with h=1.

Creates 3 plots (one per method), each with subplots for different z values.
"""

import sys
import os
import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from polyagamma import random_polyagamma

# Add src directory to path to import JAX implementation
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from polyagamma_jax import (
    sample_pg_devroye_single,
    sample_pg_saddle_single,
    sample_pg_normal_single,
)

# Configuration
N_SAMPLES = 10000
H_VALUE = 1.0
Z_VALUES = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
SEED = 42

# Method names and functions
METHODS = {
    'Devroye': sample_pg_devroye_single,
    'Saddle': sample_pg_saddle_single,
    'Normal': sample_pg_normal_single,
}


def sample_c_backend(h, z, n_samples):
    """Sample using C-backend implementation."""
    print(f"  C-backend sampling...")
    return np.array([random_polyagamma(h, z) for _ in range(n_samples)])


def sample_jax_method_batch(key, h, z, n_samples, method_func):
    """Sample using a specific JAX method with vectorization."""
    print(f"  JAX sampling...")
    # Create array of z values (all the same)
    z_array = jnp.full(n_samples, z)

    # Split keys for each sample
    keys = jax.random.split(key, n_samples)

    # Vectorize the sampling function
    sample_fn = lambda k: method_func(k, h, z)
    samples = jax.vmap(sample_fn)(keys)

    return np.array(samples)


def create_comparison_plot(method_name, method_func, z_values, h, n_samples, seed):
    """Create comparison plot for a specific method across different z values."""
    n_z = len(z_values)
    n_cols = 3
    n_rows = (n_z + n_cols - 1) // n_cols  # Ceiling division

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
    fig.suptitle(f'{method_name} Method: C-backend vs JAX (h={h})',
                 fontsize=16, fontweight='bold')

    # Flatten axes for easier indexing
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    axes_flat = axes.flatten()

    # First pass: collect all samples and compute global x-range
    all_c_samples = []
    all_jax_samples = []

    print(f"\n{'='*60}")
    print(f"Sampling for {method_name} Method (h={h})")
    print(f"{'='*60}")

    key = jax.random.PRNGKey(seed)

    for z in z_values:
        print(f"\nSampling for z={z}...")

        # Sample from C-backend
        c_samples = sample_c_backend(h, z, n_samples)
        all_c_samples.append(c_samples)

        # Sample from JAX method
        subkey = jax.random.fold_in(key, int(z * 1000))
        jax_samples = sample_jax_method_batch(subkey, h, z, n_samples, method_func)
        all_jax_samples.append(jax_samples)

        print(f"  C-backend   - Mean: {c_samples.mean():.6f}, Std: {c_samples.std():.6f}")
        print(f"  JAX ({method_name}) - Mean: {jax_samples.mean():.6f}, Std: {jax_samples.std():.6f}")

    # Compute global x-range for aligned axes
    global_min = min(np.min(c) for c in all_c_samples)
    global_max = max(np.max(c) for c in all_c_samples)
    global_min = min(global_min, min(np.min(j) for j in all_jax_samples))
    global_max = max(global_max, max(np.max(j) for j in all_jax_samples))
    margin = (global_max - global_min) * 0.05
    x_range = (global_min - margin, global_max + margin)

    print(f"\nGlobal x-range: [{x_range[0]:.4f}, {x_range[1]:.4f}]")

    # Second pass: create plots with aligned x-axis
    for idx, z in enumerate(z_values):
        ax = axes_flat[idx]
        c_samples = all_c_samples[idx]
        jax_samples = all_jax_samples[idx]

        # Compute statistics
        c_mean = c_samples.mean()
        c_std = c_samples.std()
        jax_mean = jax_samples.mean()
        jax_std = jax_samples.std()

        # Create histogram with common bins
        bins = np.linspace(x_range[0], x_range[1], 50)

        ax.hist(c_samples, bins=bins, alpha=0.5, label='C-backend',
                color='blue', density=True, edgecolor='black', linewidth=0.5)
        ax.hist(jax_samples, bins=bins, alpha=0.5, label=f'JAX ({method_name})',
                color='red', density=True, edgecolor='black', linewidth=0.5)

        # Set common x-axis range
        ax.set_xlim(x_range)

        # Add labels and title
        ax.set_xlabel('Sample Value')
        ax.set_ylabel('Density')
        ax.set_title(f'z = {z}\n' +
                    f'C: μ={c_mean:.4f}, σ={c_std:.4f}\n' +
                    f'JAX: μ={jax_mean:.4f}, σ={jax_std:.4f}',
                    fontsize=10)
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for idx in range(n_z, len(axes_flat)):
        axes_flat[idx].axis('off')

    plt.tight_layout()
    return fig


def main():
    """Main function to create all comparison plots."""
    print(f"\n{'#'*60}")
    print(f"# Distribution Comparison: C-backend vs JAX Methods")
    print(f"# h = {H_VALUE}, n_samples = {N_SAMPLES}")
    print(f"# z values: {Z_VALUES}")
    print(f"{'#'*60}")

    # Create comparison plot for each method
    figures = {}
    for method_name, method_func in METHODS.items():
        fig = create_comparison_plot(
            method_name,
            method_func,
            Z_VALUES,
            H_VALUE,
            N_SAMPLES,
            SEED
        )
        figures[method_name] = fig

        # Save figure
        filename = f'distribution_comparison_{method_name.lower()}.png'
        fig.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"\nSaved: {filename}")

    print(f"\n{'#'*60}")
    print(f"# All plots saved successfully!")
    print(f"{'#'*60}\n")

    # Note: plt.show() is commented out since we're in a non-interactive environment
    # Uncomment if running interactively
    # plt.show()


if __name__ == "__main__":
    main()
