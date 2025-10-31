"""
Compare JAX Polya-Gamma sampler with C-backend implementation.
"""
import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import time
from scipy import stats
import pytest
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Import both samplers
try:
    from polyagamma import random_polyagamma
    HAS_C_BACKEND = True
except ImportError:
    HAS_C_BACKEND = False
    print("Warning: C-backend polyagamma not available")

from polyagamma_jax import sample_pg_batch, sample_pg_improved

jax.config.update("jax_enable_x64", True)


def pg_mean_exact(z):
    """Exact mean of PG(1, z): E[PG(1,z)] = tanh(z/2) / (2z)"""
    z = np.abs(z)
    if z < 1e-10:
        return 0.25  # lim_{z->0} tanh(z/2)/(2z) = 1/4
    return np.tanh(z / 2.0) / (2.0 * z)


def pg_var_exact(z):
    """Approximate variance of PG(1, z)"""
    z = np.abs(z)
    if z < 1e-10:
        return 1.0 / 24.0  # Var[PG(1, 0)]
    return (1.0 / (4.0 * z**2)) * (1.0 / np.cosh(z / 2.0)**2)


class TestPolyagammaComparison:
    """Compare JAX and C-backend PG samplers"""

    def test_single_sample_small_z(self):
        """Test single sample for small |z|"""
        if not HAS_C_BACKEND:
            pytest.skip("C-backend not available")

        z = 0.5
        n_samples = 10000

        # C-backend samples
        c_samples = np.array([random_polyagamma(1, z) for _ in range(n_samples)])

        # JAX samples
        key = jax.random.PRNGKey(42)
        z_array = jnp.full(n_samples, z)
        jax_samples = np.array(sample_pg_batch(key, z_array))

        # Compare moments
        true_mean = pg_mean_exact(z)
        true_var = pg_var_exact(z)

        c_mean, c_std = c_samples.mean(), c_samples.std()
        jax_mean, jax_std = jax_samples.mean(), jax_samples.std()

        print(f"\nz = {z:.3f}")
        print(f"True mean: {true_mean:.6f}, True std: {np.sqrt(true_var):.6f}")
        print(f"C-backend: mean={c_mean:.6f}, std={c_std:.6f}")
        print(f"JAX:       mean={jax_mean:.6f}, std={jax_std:.6f}")

        # Should be within 3 standard errors
        se_mean = c_std / np.sqrt(n_samples)
        assert abs(jax_mean - c_mean) < 3 * se_mean, f"Means differ: JAX={jax_mean}, C={c_mean}"

    def test_single_sample_large_z(self):
        """Test single sample for large |z|"""
        if not HAS_C_BACKEND:
            pytest.skip("C-backend not available")

        z = 5.0
        n_samples = 10000

        c_samples = np.array([random_polyagamma(1, z) for _ in range(n_samples)])

        key = jax.random.PRNGKey(42)
        z_array = jnp.full(n_samples, z)
        jax_samples = np.array(sample_pg_batch(key, z_array))

        true_mean = pg_mean_exact(z)
        c_mean, c_std = c_samples.mean(), c_samples.std()
        jax_mean, jax_std = jax_samples.mean(), jax_samples.std()

        print(f"\nz = {z:.3f}")
        print(f"True mean: {true_mean:.6f}")
        print(f"C-backend: mean={c_mean:.6f}, std={c_std:.6f}")
        print(f"JAX:       mean={jax_mean:.6f}, std={jax_std:.6f}")

        se_mean = c_std / np.sqrt(n_samples)
        assert abs(jax_mean - c_mean) < 3 * se_mean, f"Means differ significantly"

    def test_range_of_z_values(self):
        """Test across range of z values"""
        if not HAS_C_BACKEND:
            pytest.skip("C-backend not available")

        z_values = [0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
        n_samples = 5000

        results = []

        for z in z_values:
            # C-backend
            c_samples = np.array([random_polyagamma(1, z) for _ in range(n_samples)])

            # JAX
            key = jax.random.PRNGKey(int(z * 1000))
            z_array = jnp.full(n_samples, z)
            jax_samples = np.array(sample_pg_batch(key, z_array))

            # True moments
            true_mean = pg_mean_exact(z)
            true_std = np.sqrt(pg_var_exact(z))

            results.append({
                'z': z,
                'true_mean': true_mean,
                'true_std': true_std,
                'c_mean': c_samples.mean(),
                'c_std': c_samples.std(),
                'jax_mean': jax_samples.mean(),
                'jax_std': jax_samples.std(),
                'c_samples': c_samples,
                'jax_samples': jax_samples,
            })

        # Print comparison table
        print("\n" + "="*80)
        print(f"{'z':<8} {'True Mean':<12} {'C Mean':<12} {'JAX Mean':<12} {'Mean Diff':<12}")
        print("="*80)
        for r in results:
            diff = abs(r['c_mean'] - r['jax_mean'])
            print(f"{r['z']:<8.2f} {r['true_mean']:<12.6f} {r['c_mean']:<12.6f} "
                  f"{r['jax_mean']:<12.6f} {diff:<12.6f}")
        print("="*80)

        # Statistical test: means should be similar
        for r in results:
            se = r['c_std'] / np.sqrt(n_samples)
            assert abs(r['c_mean'] - r['jax_mean']) < 5 * se, \
                f"z={r['z']}: means differ too much"

        return results

    def test_timing_comparison(self):
        """Compare sampling speed"""
        if not HAS_C_BACKEND:
            pytest.skip("C-backend not available")

        z_values = [0.5, 2.0, 5.0]
        n_samples_list = [1000, 10000, 100000]

        print("\n" + "="*80)
        print("TIMING COMPARISON")
        print("="*80)

        for z in z_values:
            print(f"\nz = {z:.2f}")
            print(f"{'N Samples':<15} {'C-backend (ms)':<20} {'JAX (ms)':<20} {'Speedup':<10}")
            print("-"*80)

            for n in n_samples_list:
                # C-backend timing
                start = time.time()
                c_samples = np.array([random_polyagamma(1, z) for _ in range(n)])
                c_time = (time.time() - start) * 1000

                # JAX timing (with warmup)
                key = jax.random.PRNGKey(42)
                z_array = jnp.full(n, z)

                # Warmup JIT
                _ = sample_pg_batch(key, z_array[:10])

                # Actual timing
                start = time.time()
                jax_samples = sample_pg_batch(key, z_array)
                jax_samples.block_until_ready()  # Wait for GPU
                jax_time = (time.time() - start) * 1000

                speedup = c_time / jax_time

                print(f"{n:<15} {c_time:<20.2f} {jax_time:<20.2f} {speedup:<10.2f}x")

        print("="*80)

    def test_batch_sampling(self):
        """Test batched sampling with different z values"""
        if not HAS_C_BACKEND:
            pytest.skip("C-backend not available")

        # Create array of different z values
        z_array = np.array([0.1, 0.5, 1.0, 2.0, 5.0] * 1000)  # 5000 samples
        n = len(z_array)

        # C-backend (loop)
        start = time.time()
        c_samples = np.array([random_polyagamma(1, z) for z in z_array])
        c_time = time.time() - start

        # JAX (vectorized)
        key = jax.random.PRNGKey(42)
        z_jax = jnp.array(z_array)

        # Warmup
        _ = sample_pg_batch(key, z_jax[:10])

        start = time.time()
        jax_samples = np.array(sample_pg_batch(key, z_jax))
        jax_time = time.time() - start

        print(f"\nBatch sampling ({n} samples with varying z)")
        print(f"C-backend: {c_time*1000:.2f} ms")
        print(f"JAX:       {jax_time*1000:.2f} ms")
        print(f"Speedup:   {c_time/jax_time:.2f}x")

        # Compare distributions for each z value
        unique_z = np.unique(z_array)
        for z in unique_z:
            mask = z_array == z
            c_mean = c_samples[mask].mean()
            jax_mean = jax_samples[mask].mean()
            true_mean = pg_mean_exact(z)

            print(f"z={z:.1f}: true={true_mean:.6f}, C={c_mean:.6f}, JAX={jax_mean:.6f}")

    def test_kolmogorov_smirnov(self):
        """KS test to compare distributions"""
        if not HAS_C_BACKEND:
            pytest.skip("C-backend not available")

        z_values = [0.5, 2.0, 5.0]
        n_samples = 5000

        print("\n" + "="*80)
        print("KOLMOGOROV-SMIRNOV TEST (comparing distributions)")
        print("="*80)

        for z in z_values:
            c_samples = np.array([random_polyagamma(1, z) for _ in range(n_samples)])

            key = jax.random.PRNGKey(int(z * 1000))
            z_array = jnp.full(n_samples, z)
            jax_samples = np.array(sample_pg_batch(key, z_array))

            # KS test
            ks_stat, p_value = stats.ks_2samp(c_samples, jax_samples)

            print(f"z = {z:.2f}: KS statistic = {ks_stat:.6f}, p-value = {p_value:.4f}")

            # Should not reject null hypothesis (distributions are same)
            assert p_value > 0.01, f"Distributions differ significantly for z={z}"

    def test_symmetry_negative_z(self):
        """Test that PG(1, z) = PG(1, -z)"""
        z_pos = 2.0
        z_neg = -2.0
        n_samples = 5000

        key = jax.random.PRNGKey(42)
        key1, key2 = jax.random.split(key)

        samples_pos = np.array(sample_pg_batch(key1, jnp.full(n_samples, z_pos)))
        samples_neg = np.array(sample_pg_batch(key2, jnp.full(n_samples, z_neg)))

        mean_pos = samples_pos.mean()
        mean_neg = samples_neg.mean()

        print(f"\nSymmetry test:")
        print(f"Mean for z={z_pos}: {mean_pos:.6f}")
        print(f"Mean for z={z_neg}: {mean_neg:.6f}")
        print(f"Difference: {abs(mean_pos - mean_neg):.6f}")

        # Means should be very close
        se = samples_pos.std() / np.sqrt(n_samples)
        assert abs(mean_pos - mean_neg) < 3 * se, "Symmetry violated"


def plot_comparison_distributions(results=None):
    """
    Visual comparison of distributions.

    Parameters
    ----------
    results : list of dict, optional
        Results from test_range_of_z_values()
    """
    if results is None:
        # Generate results
        if not HAS_C_BACKEND:
            print("C-backend not available, cannot plot comparison")
            return

        test = TestPolyagammaComparison()
        results = test.test_range_of_z_values()

    n_plots = len(results)
    fig, axes = plt.subplots(2, (n_plots + 1) // 2, figsize=(15, 8))
    axes = axes.ravel()

    for i, r in enumerate(results):
        ax = axes[i]

        # Plot histograms
        bins = np.linspace(0, max(r['c_samples'].max(), r['jax_samples'].max()), 50)
        ax.hist(r['c_samples'], bins=bins, alpha=0.5, density=True,
               label='C-backend', color='blue')
        ax.hist(r['jax_samples'], bins=bins, alpha=0.5, density=True,
               label='JAX', color='red')

        # Mark true mean
        ax.axvline(r['true_mean'], color='green', linestyle='--',
                  linewidth=2, label=f"True mean={r['true_mean']:.4f}")

        ax.set_xlabel('ω')
        ax.set_ylabel('Density')
        ax.set_title(f"PG(1, {r['z']:.2f})")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    # Hide unused subplots
    for i in range(len(results), len(axes)):
        axes[i].set_visible(False)

    plt.tight_layout()
    plt.savefig('polyagamma_comparison.png', dpi=150, bbox_inches='tight')
    print("\nSaved comparison plot to 'polyagamma_comparison.png'")
    plt.show()


def plot_qq_comparison():
    """Q-Q plots comparing C-backend and JAX samplers"""
    if not HAS_C_BACKEND:
        print("C-backend not available")
        return

    z_values = [0.5, 2.0, 5.0, 10.0]
    n_samples = 5000

    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    axes = axes.ravel()

    for i, z in enumerate(z_values):
        ax = axes[i]

        # Generate samples
        c_samples = np.array([random_polyagamma(1, z) for _ in range(n_samples)])

        key = jax.random.PRNGKey(int(z * 1000))
        z_array = jnp.full(n_samples, z)
        jax_samples = np.array(sample_pg_batch(key, z_array))

        # Sort for Q-Q plot
        c_sorted = np.sort(c_samples)
        jax_sorted = np.sort(jax_samples)

        # Q-Q plot
        ax.scatter(c_sorted, jax_sorted, alpha=0.3, s=1)

        # 45-degree line
        min_val = min(c_sorted.min(), jax_sorted.min())
        max_val = max(c_sorted.max(), jax_sorted.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='y=x')

        ax.set_xlabel('C-backend quantiles')
        ax.set_ylabel('JAX quantiles')
        ax.set_title(f'Q-Q Plot: PG(1, {z:.1f})')
        ax.legend()
        ax.grid(alpha=0.3)
        ax.set_aspect('equal')

    plt.tight_layout()
    plt.savefig('polyagamma_qq_plots.png', dpi=150, bbox_inches='tight')
    print("\nSaved Q-Q plots to 'polyagamma_qq_plots.png'")
    plt.show()


if __name__ == "__main__":
    print("Running Polya-Gamma sampler comparison tests...")
    print("="*80)

    test = TestPolyagammaComparison()

    # Run all tests
    try:
        print("\n1. Testing small z values...")
        test.test_single_sample_small_z()

        print("\n2. Testing large z values...")
        test.test_single_sample_large_z()

        print("\n3. Testing range of z values...")
        results = test.test_range_of_z_values()

        print("\n4. Timing comparison...")
        test.test_timing_comparison()

        print("\n5. Batch sampling test...")
        test.test_batch_sampling()

        print("\n6. Kolmogorov-Smirnov test...")
        test.test_kolmogorov_smirnov()

        print("\n7. Symmetry test...")
        test.test_symmetry_negative_z()

        print("\n" + "="*80)
        print("ALL TESTS PASSED!")
        print("="*80)

        # Generate plots
        print("\nGenerating comparison plots...")
        plot_comparison_distributions(results)
        plot_qq_comparison()

    except Exception as e:
        print(f"\nTest failed with error: {e}")
        import traceback
        traceback.print_exc()
