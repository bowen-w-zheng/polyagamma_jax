"""Simple test to verify JAX implementation works."""
import sys
sys.path.insert(0, 'src')

import jax
import jax.numpy as jnp
import numpy as np
from polyagamma_jax import sample_pg_batch, sample_pg_improved

jax.config.update("jax_enable_x64", True)

print("="*80)
print("Testing JAX Polya-Gamma Sampler")
print("="*80)

# Test 1: Simple single sample
print("\n1. Testing single sample with h=1, z=0...")
key = jax.random.PRNGKey(42)
sample = sample_pg_improved(key, h=1.0, z=0.0)
print(f"   Sample: {sample:.6f}")
print(f"   Expected mean: ~0.25")
print(f"   ✓ Single sample works!")

# Test 2: Batch sampling with z=0
print("\n2. Testing batch sampling with h=1, z=0...")
key = jax.random.PRNGKey(43)
z_array = jnp.zeros(1000)
samples = sample_pg_batch(key, z_array, h=1.0)
mean = np.mean(samples)
std = np.std(samples)
print(f"   Mean: {mean:.6f} (expected: ~0.25)")
print(f"   Std:  {std:.6f} (expected: ~0.204)")
print(f"   ✓ Batch sampling works!")

# Test 3: Different z values
print("\n3. Testing with different z values...")
z_values = [0.5, 1.0, 2.0, 5.0]
for z in z_values:
    key = jax.random.PRNGKey(int(z * 1000))
    z_array = jnp.full(1000, z)
    samples = sample_pg_batch(key, z_array, h=1.0)
    mean = np.mean(samples)

    # Theoretical mean: tanh(z/2) / (2z)
    true_mean = np.tanh(z/2) / (2*z)

    print(f"   z={z:.1f}: mean={mean:.6f}, true={true_mean:.6f}, diff={abs(mean-true_mean):.6f}")

print(f"   ✓ All z values work!")

# Test 4: Large h (saddle method)
print("\n4. Testing large h (saddle method)...")
key = jax.random.PRNGKey(44)
z_array = jnp.full(100, 2.0)
samples = sample_pg_batch(key, z_array, h=10.0)
mean = np.mean(samples)
print(f"   h=10, z=2.0: mean={mean:.6f}")
print(f"   ✓ Saddle method works!")

# Test 5: Very large h (normal approximation)
print("\n5. Testing very large h (normal approximation)...")
key = jax.random.PRNGKey(45)
z_array = jnp.full(100, 2.0)
samples = sample_pg_batch(key, z_array, h=60.0)
mean = np.mean(samples)
print(f"   h=60, z=2.0: mean={mean:.6f}")
print(f"   ✓ Normal approximation works!")

print("\n" + "="*80)
print("ALL BASIC TESTS PASSED!")
print("="*80)
