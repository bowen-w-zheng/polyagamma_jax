"""Quick test to verify the bug fixes."""

import sys
import os
import numpy as np
import jax
import jax.numpy as jnp

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from polyagamma_jax import (
    sample_pg_devroye_single,
    sample_pg_normal_single,
)

print("Testing JAX Polya-Gamma fixes...")
print("=" * 60)

# Test 1: Normal method should not produce negative values
print("\nTest 1: Normal method (h=100, z=10) - checking for negative values")
key = jax.random.PRNGKey(42)
samples = []
for i in range(100):
    key, subkey = jax.random.split(key)
    sample = sample_pg_normal_single(subkey, 100.0, 10.0)
    samples.append(float(sample))

samples = np.array(samples)
print(f"  Min: {samples.min():.6f}")
print(f"  Max: {samples.max():.6f}")
print(f"  Mean: {samples.mean():.6f}")
print(f"  Negative values: {np.sum(samples < 0)}")
if np.any(samples < 0):
    print("  ❌ FAIL: Found negative values!")
else:
    print("  ✓ PASS: No negative values")

# Test 2: Devroye method for small z should not produce zeros
print("\nTest 2: Devroye method (h=1, z=0.1) - checking for zeros")
key = jax.random.PRNGKey(43)
samples = []
for i in range(100):
    key, subkey = jax.random.split(key)
    sample = sample_pg_devroye_single(subkey, 1.0, 0.1)
    samples.append(float(sample))

samples = np.array(samples)
print(f"  Min: {samples.min():.6f}")
print(f"  Max: {samples.max():.6f}")
print(f"  Mean: {samples.mean():.6f}")
print(f"  Zero values: {np.sum(samples == 0.0)}")
print(f"  Very small (<0.001): {np.sum(samples < 0.001)}")

if np.sum(samples == 0.0) > 10:
    print("  ❌ FAIL: Too many zeros!")
elif np.sum(samples < 0.001) > 20:
    print("  ⚠ WARNING: Many very small values")
else:
    print("  ✓ PASS: Distribution looks reasonable")

# Test 3: Devroye method for z=0.5
print("\nTest 3: Devroye method (h=1, z=0.5)")
key = jax.random.PRNGKey(44)
samples = []
for i in range(100):
    key, subkey = jax.random.split(key)
    sample = sample_pg_devroye_single(subkey, 1.0, 0.5)
    samples.append(float(sample))

samples = np.array(samples)
print(f"  Min: {samples.min():.6f}")
print(f"  Max: {samples.max():.6f}")
print(f"  Mean: {samples.mean():.6f}")
print(f"  Zero values: {np.sum(samples == 0.0)}")

# Test 4: Devroye method for z=2.0
print("\nTest 4: Devroye method (h=1, z=2.0)")
key = jax.random.PRNGKey(45)
samples = []
for i in range(100):
    key, subkey = jax.random.split(key)
    sample = sample_pg_devroye_single(subkey, 1.0, 2.0)
    samples.append(float(sample))

samples = np.array(samples)
print(f"  Min: {samples.min():.6f}")
print(f"  Max: {samples.max():.6f}")
print(f"  Mean: {samples.mean():.6f}")
print(f"  Zero values: {np.sum(samples == 0.0)}")

print("\n" + "=" * 60)
print("Tests completed!")
