"""Quick comparison test with C backend."""
import sys
sys.path.insert(0, 'src')

import numpy as np
import jax
import jax.numpy as jnp

try:
    from polyagamma import random_polyagamma
    HAS_C_BACKEND = True
except ImportError:
    HAS_C_BACKEND = False
    print("Warning: C-backend not available")
    sys.exit(0)

from polyagamma_jax import sample_pg_batch

jax.config.update("jax_enable_x64", True)

print("="*80)
print("Quick JAX vs C-backend Comparison")
print("="*80)

def pg_mean_exact(z):
    """Exact mean of PG(1, z)"""
    z = np.abs(z)
    if z < 1e-10:
        return 0.25
    return np.tanh(z / 2.0) / (2.0 * z)

# Test a few key z values
z_values = [0.0, 0.5, 2.0, 5.0]
n_samples = 2000

print("\nTesting different z values (n=2000 each):")
print(f"{'z':<8} {'True Mean':<12} {'C Mean':<12} {'JAX Mean':<12} {'Diff':<12}")
print("-"*80)

all_pass = True
for z in z_values:
    # C-backend
    c_samples = np.array([random_polyagamma(1, z) for _ in range(n_samples)])
    c_mean = c_samples.mean()
    c_std = c_samples.std()

    # JAX
    key = jax.random.PRNGKey(int(z * 1000))
    z_array = jnp.full(n_samples, z)
    jax_samples = np.array(sample_pg_batch(key, z_array))
    jax_mean = jax_samples.mean()

    # True mean
    true_mean = pg_mean_exact(z)

    # Check if within reasonable bounds (3 standard errors)
    se = c_std / np.sqrt(n_samples)
    diff = abs(c_mean - jax_mean)
    passed = diff < 5 * se

    status = "✓" if passed else "✗"
    print(f"{z:<8.1f} {true_mean:<12.6f} {c_mean:<12.6f} {jax_mean:<12.6f} {diff:<12.6f} {status}")

    if not passed:
        all_pass = False

print("-"*80)

if all_pass:
    print("\n✓ All tests passed! JAX implementation matches C backend.")
else:
    print("\n✗ Some tests failed. There may be numerical differences.")

print("\n" + "="*80)
