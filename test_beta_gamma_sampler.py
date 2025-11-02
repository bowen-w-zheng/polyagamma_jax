#!/usr/bin/env python
"""
Test script for beta_gamma_pg_fixed_latents_joint_jax.py
Verifies that both mean approximation and exact PG sampling work correctly.
"""
import numpy as np
import jax.numpy as jnp

def test_samplers():
    print("=" * 80)
    print("Testing Beta-Gamma Gibbs Sampler with Mean vs Exact PG Sampling")
    print("=" * 80)

    # Import the sampler
    from src.beta_gamma_pg_fixed_latents_joint_jax import (
        sample_beta_gamma_from_fixed_latents_joint,
        JointSamplerConfig,
        _HAS_PG
    )

    print(f"\nPolya-Gamma sampler available: {_HAS_PG}")

    # Create synthetic data
    np.random.seed(42)
    R = 2   # Number of spatial locations
    S = 3   # Number of units
    T = 100 # Number of time points
    J = 4   # Number of frequency bands
    K = 10  # Number of centers

    # Generate random spikes
    spikes = np.random.binomial(1, 0.3, size=(R, S, T)).astype(np.uint8)

    # Generate random EM parameters
    X_mean = np.random.randn(J, 2, K) + 1j * np.random.randn(J, 2, K)
    D_mean = np.random.randn(R, J, 2) + 1j * np.random.randn(R, J, 2)
    freqs_hz = np.array([4.0, 8.0, 13.0, 30.0])
    centres_sec = np.linspace(0, 1.0, K)
    delta_spk = 0.01

    # Generate history features
    H_hist = np.random.randn(S, T, 5)  # 5 history lags

    print(f"\nData shapes:")
    print(f"  Spikes: {spikes.shape}")
    print(f"  History: {H_hist.shape}")
    print(f"  X_mean: {X_mean.shape}")
    print(f"  D_mean: {D_mean.shape}")

    # Test 1: Mean approximation (should always work)
    print("\n" + "-" * 80)
    print("Test 1: Mean Approximation (use_pg_sampler=False)")
    print("-" * 80)

    cfg_mean = JointSamplerConfig(
        n_warmup=50,
        n_samples=20,
        thin=2,
        use_pg_sampler=False,
        omega_floor=1e-6,
        tau2_intercept=1e4,
        tau2_beta=1e2,
        tau2_gamma=625.0,
        standardize_reim=False,
        standardize_hist=False,
        use_ard_beta=False,
        verbose=True,
        rng=np.random.default_rng(42)
    )

    trace_mean = sample_beta_gamma_from_fixed_latents_joint(
        spikes=spikes,
        H_hist=H_hist,
        X_mean=X_mean,
        D_mean=D_mean,
        freqs_hz=freqs_hz,
        centres_sec=centres_sec,
        delta_spk=delta_spk,
        bands_idx=None,
        cfg=cfg_mean
    )

    print(f"\nResults (Mean Approximation):")
    print(f"  Beta shape: {trace_mean.beta.shape}")
    print(f"  Gamma shape: {trace_mean.gamma.shape if trace_mean.gamma is not None else 'None'}")
    print(f"  Beta mean: {trace_mean.beta.mean(axis=(0, 1)):.4f}")
    print(f"  Beta std: {trace_mean.beta.std(axis=(0, 1)):.4f}")
    if trace_mean.gamma is not None:
        print(f"  Gamma mean: {trace_mean.gamma.mean(axis=(0, 1)):.4f}")
        print(f"  Gamma std: {trace_mean.gamma.std(axis=(0, 1)):.4f}")
    print(f"  Metadata: {trace_mean.meta}")

    # Test 2: Exact PG sampling (only if available)
    if _HAS_PG:
        print("\n" + "-" * 80)
        print("Test 2: Exact Polya-Gamma Sampling (use_pg_sampler=True)")
        print("-" * 80)

        cfg_pg = JointSamplerConfig(
            n_warmup=50,
            n_samples=20,
            thin=2,
            use_pg_sampler=True,  # Use exact PG sampling
            omega_floor=1e-6,
            tau2_intercept=1e4,
            tau2_beta=1e2,
            tau2_gamma=625.0,
            standardize_reim=False,
            standardize_hist=False,
            use_ard_beta=False,
            verbose=True,
            rng=np.random.default_rng(42)
        )

        trace_pg = sample_beta_gamma_from_fixed_latents_joint(
            spikes=spikes,
            H_hist=H_hist,
            X_mean=X_mean,
            D_mean=D_mean,
            freqs_hz=freqs_hz,
            centres_sec=centres_sec,
            delta_spk=delta_spk,
            bands_idx=None,
            cfg=cfg_pg
        )

        print(f"\nResults (Exact PG Sampling):")
        print(f"  Beta shape: {trace_pg.beta.shape}")
        print(f"  Gamma shape: {trace_pg.gamma.shape if trace_pg.gamma is not None else 'None'}")
        print(f"  Beta mean: {trace_pg.beta.mean(axis=(0, 1)):.4f}")
        print(f"  Beta std: {trace_pg.beta.std(axis=(0, 1)):.4f}")
        if trace_pg.gamma is not None:
            print(f"  Gamma mean: {trace_pg.gamma.mean(axis=(0, 1)):.4f}")
            print(f"  Gamma std: {trace_pg.gamma.std(axis=(0, 1)):.4f}")
        print(f"  Metadata: {trace_pg.meta}")

        # Compare results
        print("\n" + "-" * 80)
        print("Comparison: Mean Approximation vs Exact PG Sampling")
        print("-" * 80)
        print(f"  Beta mean difference: {np.abs(trace_mean.beta.mean() - trace_pg.beta.mean()):.6f}")
        print(f"  Beta std difference: {np.abs(trace_mean.beta.std() - trace_pg.beta.std()):.6f}")
        if trace_mean.gamma is not None and trace_pg.gamma is not None:
            print(f"  Gamma mean difference: {np.abs(trace_mean.gamma.mean() - trace_pg.gamma.mean()):.6f}")
            print(f"  Gamma std difference: {np.abs(trace_mean.gamma.std() - trace_pg.gamma.std()):.6f}")

        print("\n✓ Both sampling methods completed successfully!")
    else:
        print("\n" + "-" * 80)
        print("Test 2: Skipped (Polya-Gamma sampler not available)")
        print("-" * 80)
        print("  To test exact PG sampling, ensure polyagamma_jax is properly installed.")

    print("\n" + "=" * 80)
    print("All tests completed!")
    print("=" * 80)

    return trace_mean, trace_pg if _HAS_PG else None


if __name__ == "__main__":
    trace_mean, trace_pg = test_samplers()
