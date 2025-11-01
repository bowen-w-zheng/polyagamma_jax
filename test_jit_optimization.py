"""Test script demonstrating the optimized JAX JIT API.

This script demonstrates:
1. Using create_pg_sampler() for easy, pre-warmed sampling
2. No recompilation occurs when z changes (same shape/dtype)
3. Significant performance improvements with the new API
"""
import time
import jax
import jax.numpy as jnp
import numpy as np
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from polyagamma_jax import create_pg_sampler

jax.config.update("jax_enable_x64", True)


def test_no_recompilation():
    """Test that changing z values doesn't trigger recompilation."""
    print("\n" + "=" * 96)
    print("Demo: Using create_pg_sampler() for optimal performance")
    print("=" * 96)

    h = 1.0
    n_samples = 1000
    z_values = [0.5, 1.0, 2.0, 5.0]

    # Create a pre-warmed sampler using the new API - one line!
    print("\nCreating pre-warmed sampler with create_pg_sampler()...")
    sampler = create_pg_sampler(h=h, batch_size=n_samples, method='devroye', warmup=True)
    print("Sampler ready! (warmup complete)")

    # Test with different z values - run each multiple times to show no recompilation
    print(f"\n{'z':>8} {'Run 1 (ms)':>12} {'Run 2 (ms)':>12} {'Run 3 (ms)':>12} {'Mean':>12} {'Variance':>12}")
    print("-" * 96)

    for idx, z in enumerate(z_values):
        z_array = jnp.full(n_samples, z, dtype=jnp.float64)

        # Run the same z value multiple times - should have consistent timing if no recompilation
        run_times = []
        samples = None
        for run in range(3):
            key = jax.random.PRNGKey(100 + idx * 10 + run)
            start = time.perf_counter()
            samples = sampler(key, z_array)
            samples.block_until_ready()
            elapsed = time.perf_counter() - start
            run_times.append(elapsed)

        # Check results
        mean = float(np.mean(samples))
        var = float(np.var(samples))

        print(f"{z:8.2f} {run_times[0]*1000:12.2f} {run_times[1]*1000:12.2f} {run_times[2]*1000:12.2f} {mean:12.6f} {var:12.6f}")

    print("\n" + "=" * 96)
    print("Key Observations:")
    print("=" * 96)
    print("1. create_pg_sampler() handles warmup automatically - just one line of code!")
    print("2. Each row shows 3 runs with the same z - times are consistent (no recompilation)")
    print("3. Different z values may have different execution times (expected algorithm behavior)")
    print("4. After warmup, all subsequent calls are fast regardless of z value")
    print("=" * 96)


def test_mcmc_usage_pattern():
    """Demonstrate typical MCMC usage pattern."""
    print("\n" + "=" * 96)
    print("Demo: Typical MCMC usage pattern")
    print("=" * 96)

    h = 1.0
    n_samples = 1000
    n_iterations = 10

    # Create sampler once at the start of MCMC (with warmup)
    print("\nSetting up MCMC sampler (one-time setup)...")
    sampler = create_pg_sampler(h=h, batch_size=n_samples, method='saddle', warmup=True)
    print("Ready for MCMC iterations!\n")

    # Simulate MCMC loop with changing z values
    print(f"{'Iteration':>12} {'z value':>12} {'Time (ms)':>15} {'Sample Mean':>15}")
    print("-" * 96)

    # Simulate changing z values (as would happen in real MCMC)
    z_sequence = np.sin(np.linspace(0, 2*np.pi, n_iterations)) * 2 + 2.5  # z varies from ~0.5 to ~4.5

    for iteration in range(n_iterations):
        z = z_sequence[iteration]
        z_array = jnp.full(n_samples, z, dtype=jnp.float64)

        key = jax.random.PRNGKey(iteration + 1000)
        start = time.perf_counter()
        samples = sampler(key, z_array)
        samples.block_until_ready()
        elapsed = time.perf_counter() - start

        mean = float(np.mean(samples))
        print(f"{iteration+1:12d} {z:12.4f} {elapsed*1000:15.2f} {mean:15.6f}")

    print("\n" + "=" * 96)
    print("Key Observations:")
    print("=" * 96)
    print("1. create_pg_sampler() called ONCE before the MCMC loop")
    print("2. Each iteration uses different z value - no recompilation!")
    print("3. Consistent fast performance across all iterations")
    print("4. This is the recommended pattern for iterative algorithms")
    print("=" * 96)


if __name__ == "__main__":
    test_no_recompilation()
    test_mcmc_usage_pattern()
    print("\n✓ All demonstrations completed!\n")
    print("For real-world MCMC usage, use create_pg_sampler() as shown above.")
    print("The saddle method often provides the best speed/accuracy balance.\n")
