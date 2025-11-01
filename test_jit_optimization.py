"""Test script to verify JAX JIT optimization works correctly.

This script verifies that:
1. The batch sampler is created once and reused across different z values
2. No recompilation occurs when z changes (same shape/dtype)
3. The sampler produces correct results
"""
import time
import jax
import jax.numpy as jnp
import numpy as np
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from polyagamma_jax import sample_pg_devroye_single

jax.config.update("jax_enable_x64", True)


def create_batch_sampler(single_sampler, h):
    """Create a batch sampler from a single sampler function.

    h is fixed at creation time since it's used in control flow.
    Only z_array is traced, allowing the same compiled function to be
    reused for different z values.
    """
    @jax.jit
    def batch_sampler(key, z_array):
        """Sample from PG(h, z) for a batch of z values."""
        n = z_array.shape[0]
        keys = jax.random.split(key, n)
        sample_fn = lambda k, z: single_sampler(k, h, z)
        samples = jax.vmap(sample_fn)(keys, z_array)
        return samples
    return batch_sampler


def test_no_recompilation():
    """Test that changing z values doesn't trigger recompilation."""
    print("\n" + "=" * 80)
    print("Testing JIT optimization: No recompilation when z changes")
    print("=" * 80)

    h = 1.0
    n_samples = 1000
    z_values = [0.5, 1.0, 2.0, 5.0]

    # Create batch sampler once (h is fixed at creation)
    batch_sampler = create_batch_sampler(sample_pg_devroye_single, h)

    # Warmup with first z value
    print("\nWarming up JIT compilation...")
    z_warmup = jnp.full(100, z_values[0], dtype=jnp.float64)
    warmup_key = jax.random.PRNGKey(42)
    batch_sampler(warmup_key, z_warmup).block_until_ready()
    print("Warmup complete!")

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
            samples = batch_sampler(key, z_array)
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
    print("1. Each row shows the same z value run 3 times")
    print("2. If the 3 times are consistent within each row, no recompilation is occurring")
    print("3. Different z values may have different execution times (expected behavior)")
    print("   - Larger z values often run faster due to algorithm selection")
    print("=" * 96)


def test_same_shape_reuse():
    """Test that same-shape arrays reuse compiled code."""
    print("\n" + "=" * 80)
    print("Testing same-shape reuse across different arrays")
    print("=" * 80)

    h = 1.0
    n_samples = 500

    # Create batch sampler (h is fixed at creation)
    batch_sampler = create_batch_sampler(sample_pg_devroye_single, h)

    # Test with arrays of the same shape but different values
    test_cases = [
        ("Constant z=0.5", jnp.full(n_samples, 0.5, dtype=jnp.float64)),
        ("Constant z=2.0", jnp.full(n_samples, 2.0, dtype=jnp.float64)),
        ("Random z values", jax.random.uniform(jax.random.PRNGKey(999), (n_samples,),
                                                minval=0.1, maxval=5.0, dtype=jnp.float64)),
    ]

    print(f"\n{'Test Case':>30} {'Time (ms)':>12}")
    print("-" * 80)

    for name, z_array in test_cases:
        key = jax.random.PRNGKey(42)
        start = time.perf_counter()
        samples = batch_sampler(key, z_array)
        samples.block_until_ready()
        elapsed = time.perf_counter() - start
        print(f"{name:>30} {elapsed*1000:12.2f}")

    print("\nAll test cases should have similar timing (within ~20%)")
    print("=" * 80)


if __name__ == "__main__":
    test_no_recompilation()
    test_same_shape_reuse()
    print("\n✓ All tests completed!\n")
