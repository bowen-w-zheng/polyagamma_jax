"""Standalone comparison between C-backed and JAX Polya-Gamma samplers.

Run this module as a script to generate summary tables comparing
per-sample mean, variance, and runtime for the available sampling
methods in :mod:`polyagamma_jax` against the reference C backend.
"""
from __future__ import annotations

import os
import sys
import time
from typing import Iterable, Tuple

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
if REPO_ROOT in sys.path:
    sys.path.remove(REPO_ROOT)

try:
    from polyagamma import random_polyagamma
except ImportError as exc:  # pragma: no cover - optional dependency in some envs
    raise SystemExit(
        "The C-backed `polyagamma` package is required to run this comparison.\n"
        "Install it with: pip install polyagamma"
    ) from exc

jax.config.update("jax_enable_x64", True)


def _sample_c_backend(h: float, z: float, n_samples: int) -> Tuple[np.ndarray, float]:
    """Return ``n_samples`` draws from the C implementation and elapsed seconds."""
    start = time.perf_counter()
    samples = np.array([random_polyagamma(h, z) for _ in range(n_samples)])
    elapsed = time.perf_counter() - start
    return samples, elapsed


def compare_methods(
    h: float = 1.0,
    z_values: Iterable[float] = (0.1, 0.5, 2.0, 10.0),
    n_samples: int = 10000,
) -> None:
    """Print comparison tables for JAX Polya-Gamma samplers vs C-backend.

    This function compares the C-backend implementation (which uses its own
    hybrid method) against individual JAX sampling methods.

    Parameters
    ----------
    h : float, optional
        Shape parameter for PG(h, z) distribution (default: 1.0)
    z_values : iterable of float, optional
        Values of z parameter to test (default: (0.1, 0.5, 2.0, 10.0))
    n_samples : int, optional
        Number of samples to draw for each test (default: 10000)
    """
    from polyagamma_jax import create_pg_sampler

    # JAX methods to compare
    methods = (
        ("devroye", "devroye"),
        ("saddle", "saddle"),
        ("normal", "normal"),
        ("hybrid", "hybrid"),
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
    print(f"POLYA-GAMMA SAMPLER COMPARISON: C-backend (hybrid) vs JAX methods")
    print(f"Parameters: h={h}, n_samples={n_samples}")
    print("=" * 100)
    print("\nNOTE: C-backend uses its own hybrid method for all comparisons.")
    print("      JAX methods are tested individually to show their performance characteristics.")
    print("=" * 100)

    for method_name, method in methods:
        print(f"\n{'=' * 100}")
        print(f"JAX Method: {method_name.upper()}")
        print(f"{'=' * 100}")
        print(header)
        print("-" * 100)

        # Create pre-warmed sampler using the new API
        # This ensures optimal performance with no recompilation when z changes
        batch_sampler = create_pg_sampler(h=h, batch_size=n_samples, method=method, warmup=True)

        for idx, z in enumerate(z_values):
            # Sample from C backend (uses C's hybrid method)
            c_samples, c_elapsed = _sample_c_backend(h, z, n_samples)
            c_mean = c_samples.mean()
            c_var = c_samples.var()
            c_time_ms = c_elapsed * 1000

            # Sample from JAX implementation using pre-warmed batch sampler
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
    print("Summary:")
    print("=" * 100)
    print("  - 'Speedup' shows JAX speedup relative to C-backend (higher is better)")
    print("  - JAX times are best of 5 runs after warmup (no JIT compilation overhead)")
    print("  - C-backend times are single-run measurements using C's hybrid method")
    print("  - Mean and variance should be similar across implementations")
    print("  - JAX 'hybrid' method automatically selects the best algorithm (like C does)")
    print("  - JAX 'saddle' often shows excellent speed+accuracy balance")
    print("  - Batch processing and GPU acceleration provide significant JAX speedups")
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
