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

try:
    from polyagamma import random_polyagamma
except ImportError as exc:  # pragma: no cover - optional dependency in some envs
    raise SystemExit(
        "The C-backed `polyagamma` package is required to run this comparison."
    ) from exc

from polyagamma_jax import (
    sample_pg_devroye_single,
    sample_pg_normal_single,
    sample_pg_saddle_single,
)

jax.config.update("jax_enable_x64", True)

Sampler = Callable[[jax.random.PRNGKey, float, float], jnp.ndarray]


def _sample_c_backend(h: float, z: float, n_samples: int) -> Tuple[np.ndarray, float]:
    """Return ``n_samples`` draws from the C implementation and elapsed seconds."""
    start = time.perf_counter()
    samples = np.array([random_polyagamma(h, z) for _ in range(n_samples)])
    elapsed = time.perf_counter() - start
    return samples, elapsed


def _sample_jax_method(
    single_sampler: Sampler,
    h: float,
    z: float,
    n_samples: int,
    seed: int,
) -> Tuple[np.ndarray, float]:
    """Collect ``n_samples`` draws for ``single_sampler`` and elapsed seconds."""

    def batch_sampler(keys: jnp.ndarray) -> jnp.ndarray:
        return jax.vmap(lambda key: single_sampler(key, h, z))(keys)

    batched = jax.jit(batch_sampler)

    key = jax.random.PRNGKey(seed)
    keys = jax.random.split(key, n_samples)

    warmup_size = min(32, n_samples)
    batched(keys[:warmup_size]).block_until_ready()

    start = time.perf_counter()
    samples = batched(keys)
    samples.block_until_ready()
    elapsed = time.perf_counter() - start

    return np.asarray(samples), elapsed


def compare_methods(
    h: float = 1.0,
    z_values: Iterable[float] = (0.1, 0.5, 2.0, 10.0),
    n_samples: int = 5000,
) -> None:
    """Print comparison tables for each sampler defined in :mod:`polyagamma_jax`."""

    methods = (
        ("devroye", sample_pg_devroye_single),
        ("saddle", sample_pg_saddle_single),
        ("normal", sample_pg_normal_single),
    )

    header = (
        f"{'z':>6}"
        f"{'C Mean':>12}"
        f"{'C Var':>12}"
        f"{'C Time (ms)':>14}"
        f"{'JAX Mean':>12}"
        f"{'JAX Var':>12}"
        f"{'JAX Time (ms)':>14}"
    )

    for method_name, sampler in methods:
        print("=" * 80)
        print(f"Method: {method_name.upper()} (h={h}, n={n_samples})")
        print("=" * 80)
        print(header)

        for idx, z in enumerate(z_values):
            c_samples, c_elapsed = _sample_c_backend(h, z, n_samples)
            jax_samples, jax_elapsed = _sample_jax_method(
                sampler, h, z, n_samples, seed=idx + 1
            )

            row = (
                f"{z:6.2f}"
                f"{c_samples.mean():12.6f}"
                f"{c_samples.var():12.6f}"
                f"{c_elapsed * 1000:14.2f}"
                f"{jax_samples.mean():12.6f}"
                f"{jax_samples.var():12.6f}"
                f"{jax_elapsed * 1000:14.2f}"
            )
            print(row)

        print()


if __name__ == "__main__":
    compare_methods()
