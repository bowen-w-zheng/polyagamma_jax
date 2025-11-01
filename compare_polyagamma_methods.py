"""Standalone comparison between C-backed and JAX Polya-Gamma samplers.

Run this module as a script to generate summary tables comparing
per-sample mean, variance, and runtime for the available sampling
methods in :mod:`polyagamma_jax` against the reference C backend.
"""
from __future__ import annotations

import os
import sys
import time
from typing import Callable, Iterable, Tuple

import numpy as np

try:
    import jax
    import jax.numpy as jnp
except ImportError as exc:  # pragma: no cover - jax is an optional dependency
    raise SystemExit(
        "This comparison script requires JAX to be installed."
    ) from exc

# Ensure the local source tree is importable when running from the repo root.
REPO_ROOT = os.path.abspath(os.path.dirname(__file__))
SRC_PATH = os.path.join(REPO_ROOT, "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

# Remove local polyagamma directory from sys.path to avoid shadowing
# the installed C-backend polyagamma package
import sys
if REPO_ROOT in sys.path:
    sys.path.remove(REPO_ROOT)

try:
    from polyagamma import random_polyagamma
except ImportError as exc:  # pragma: no cover - optional dependency in some envs
    raise SystemExit(
        "The C-backed `polyagamma` package is required to run this comparison.\n"
        "Install it with: pip install polyagamma"
    ) from exc

from polyagamma_jax import (
    sample_pg_devroye_single,
    sample_pg_normal_single,
    sample_pg_saddle_single,
    sample_pg_batch,
)

jax.config.update("jax_enable_x64", True)

SingleSampler = Callable[[jax.random.PRNGKey, float, float], jnp.ndarray]


def _sample_c_backend(h: float, z: float, n_samples: int) -> Tuple[np.ndarray, float]:
    """Return ``n_samples`` draws from the C implementation and elapsed seconds."""
    start = time.perf_counter()
    samples = np.array([random_polyagamma(h, z) for _ in range(n_samples)])
    elapsed = time.perf_counter() - start
    return samples, elapsed


def _create_batch_sampler(single_sampler: SingleSampler, h: float):
    """Create a batch sampler from a single sampler function.

    The returned batch sampler accepts z_array as a traced argument to avoid
    recompilation when z values change (as long as shape and dtype remain constant).
    h is fixed at creation time since it's used in control flow within the samplers.

    Parameters
    ----------
    single_sampler : callable
        Single sample function with signature (key, h, z) -> sample
    h : float
        Shape parameter (fixed at JIT compile time)

    Returns
    -------
    callable
        Batch sampler with signature (key, z_array) -> samples
    """
    @jax.jit
    def batch_sampler(key: jax.random.PRNGKey, z_array: jnp.ndarray) -> jnp.ndarray:
        """Sample from PG(h, z) for a batch of z values."""
        n = z_array.shape[0]
        keys = jax.random.split(key, n)
        sample_fn = lambda k, z: single_sampler(k, h, z)
        samples = jax.vmap(sample_fn)(keys, z_array)
        return samples

    return batch_sampler


def _sample_jax_method(
    single_sampler: SingleSampler,
    h: float,
    z: float,
    n_samples: int,
    seed: int,
) -> Tuple[np.ndarray, float]:
    """Collect ``n_samples`` draws for ``single_sampler`` and elapsed seconds.

    Uses efficient batch sampling by creating an array of identical z values
    and using vectorized sampling. The batch sampler is created once and reused
    across warmup and timed runs to avoid recompilation.
    """
    # Create batch sampler for this method (only once)
    # h is fixed at creation time, z_array is traced to avoid recompilation
    batch_sampler = _create_batch_sampler(single_sampler, h)

    # Create arrays for batch sampling
    key = jax.random.PRNGKey(seed)
    z_array = jnp.full(n_samples, z, dtype=jnp.float64)

    # Warmup: Run multiple times to ensure JIT compilation is complete
    warmup_size = min(500, n_samples)
    z_warmup = z_array[:warmup_size]
    for _ in range(3):
        batch_sampler(key, z_warmup).block_until_ready()

    # Timed sampling - run multiple iterations and take the best time
    n_iterations = 5
    times = []
    all_samples = None
    for i in range(n_iterations):
        key_i = jax.random.PRNGKey(seed + i)
        start = time.perf_counter()
        samples = batch_sampler(key_i, z_array)
        samples.block_until_ready()
        elapsed = time.perf_counter() - start
        times.append(elapsed)
        if i == 0:
            all_samples = samples

    # Use the minimum time (best performance after warmup)
    min_time = min(times)

    return np.asarray(all_samples), min_time


def compare_methods(
    h: float = 1.0,
    z_values: Iterable[float] = (0.1, 0.5, 2.0, 10.0),
    n_samples: int = 10000,
) -> None:
    """Print comparison tables for each sampler defined in :mod:`polyagamma_jax`.

    Parameters
    ----------
    h : float, optional
        Shape parameter for PG(h, z) distribution (default: 1.0)
    z_values : iterable of float, optional
        Values of z parameter to test (default: (0.1, 0.5, 2.0, 10.0))
    n_samples : int, optional
        Number of samples to draw for each test (default: 10000)
    """
    from polyagamma_jax import sample_pg_single

    methods = (
        ("devroye", sample_pg_devroye_single),
        ("saddle", sample_pg_saddle_single),
        ("normal", sample_pg_normal_single),
        ("hybrid", sample_pg_single),
    )

    header = (
        f"{'z':>8}"
        f"{'C Mean':>12}"
        f"{'C Var':>12}"
        f"{'C Time':>12}"
        f"{'JAX Mean':>12}"
        f"{'JAX Var':>12}"
        f"{'JAX Time':>12}"
        f"{'Speedup':>10}"
    )

    print("\n" + "=" * 100)
    print(f"POLYA-GAMMA SAMPLER COMPARISON: C-backend vs JAX")
    print(f"Parameters: h={h}, n_samples={n_samples}")
    print("=" * 100)

    for method_name, sampler in methods:
        print(f"\n{'=' * 100}")
        print(f"Method: {method_name.upper()}")
        print(f"{'=' * 100}")
        print(header)
        print("-" * 100)

        # Create batch sampler once for this method and reuse across all z values
        # h is fixed at creation time, only z values change (avoiding recompilation)
        batch_sampler = _create_batch_sampler(sampler, h)

        # Perform initial warmup with first z value to trigger compilation
        first_z = list(z_values)[0]
        warmup_key = jax.random.PRNGKey(42)
        warmup_size = min(500, n_samples)
        z_warmup = jnp.full(warmup_size, first_z, dtype=jnp.float64)
        for _ in range(3):
            batch_sampler(warmup_key, z_warmup).block_until_ready()

        for idx, z in enumerate(z_values):
            # Sample from C backend
            c_samples, c_elapsed = _sample_c_backend(h, z, n_samples)
            c_mean = c_samples.mean()
            c_var = c_samples.var()
            c_time_ms = c_elapsed * 1000

            # Sample from JAX implementation using pre-created batch sampler
            z_array = jnp.full(n_samples, z, dtype=jnp.float64)

            # Timed sampling - run multiple iterations and take the best time
            n_iterations = 5
            times = []
            all_samples = None
            for i in range(n_iterations):
                key_i = jax.random.PRNGKey(idx + 42 + i)
                start = time.perf_counter()
                samples = batch_sampler(key_i, z_array)
                samples.block_until_ready()
                elapsed = time.perf_counter() - start
                times.append(elapsed)
                if i == 0:
                    all_samples = samples

            jax_elapsed = min(times)
            jax_samples = np.asarray(all_samples)
            jax_mean = jax_samples.mean()
            jax_var = jax_samples.var()
            jax_time_ms = jax_elapsed * 1000

            # Compute speedup (C time / JAX time)
            speedup = c_elapsed / jax_elapsed if jax_elapsed > 0 else float('inf')

            # Print results
            row = (
                f"{z:8.2f}"
                f"{c_mean:12.6f}"
                f"{c_var:12.6f}"
                f"{c_time_ms:11.2f}ms"
                f"{jax_mean:12.6f}"
                f"{jax_var:12.6f}"
                f"{jax_time_ms:11.2f}ms"
                f"{speedup:10.2f}x"
            )
            print(row)

    print("\n" + "=" * 100)
    print("Notes:")
    print("  - 'Speedup' shows JAX speedup relative to C-backend (higher is better)")
    print("  - JAX times represent the best of 5 runs after warmup (excludes JIT compilation)")
    print("  - C-backend times are single-run measurements")
    print("  - Mean and variance should be similar between C and JAX implementations")
    print("  - Hybrid method automatically selects the best algorithm based on h and z")
    print("  - JAX performance improves significantly with larger batch sizes (GPU acceleration)")
    print("=" * 100 + "\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Compare C-backend and JAX Polya-Gamma samplers"
    )
    parser.add_argument(
        "--h", type=float, default=1.0,
        help="Shape parameter (default: 1.0)"
    )
    parser.add_argument(
        "--z-values", type=float, nargs="+", default=[0.1, 0.5, 2.0, 10.0],
        help="Z values to test (default: 0.1 0.5 2.0 10.0)"
    )
    parser.add_argument(
        "--n-samples", type=int, default=10000,
        help="Number of samples per test (default: 10000)"
    )

    args = parser.parse_args()

    compare_methods(h=args.h, z_values=args.z_values, n_samples=args.n_samples)
