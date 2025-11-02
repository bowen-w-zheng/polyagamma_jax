from __future__ import annotations
import jax
import jax.numpy as jnp
from jax import jit, vmap, lax
import numpy as np
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple, Dict, Any
from functools import partial

# Enable 64-bit precision
jax.config.update("jax_enable_x64", True)

# JAX Polya-Gamma sampler
_HAS_PG = False
try:
    from polyagamma_jax import sample_pg_single
    _HAS_PG = True
except Exception:
    _HAS_PG = False


# ─────────────────────────────────────────────────────────────────────────────
# Predictors (keep as NumPy for preprocessing)
# ─────────────────────────────────────────────────────────────────────────────
def _nearest_centres_idx(centres_sec: np.ndarray, t_sec: np.ndarray) -> np.ndarray:
    c = np.asarray(centres_sec, float).ravel()
    t = np.asarray(t_sec, float).ravel()
    ir = np.searchsorted(c, t, side="left")
    il = np.clip(ir - 1, 0, c.size - 1)
    ir = np.clip(ir, 0, c.size - 1)
    use_r = np.abs(c[ir] - t) < np.abs(c[il] - t)
    return np.where(use_r, ir, il).astype(np.int32)

def build_rotated_predictors_from_em(
    *,
    X_mean: np.ndarray,
    D_mean: np.ndarray,
    freqs_hz: np.ndarray,
    delta_spk: float,
    centres_sec: np.ndarray,
    T: int,
    bands_idx: Optional[Sequence[int]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    X = np.asarray(X_mean)
    D = np.asarray(D_mean)
    R = int(D.shape[0]); J, M, K = X.shape
    freqs = np.asarray(freqs_hz, float).reshape(J,)

    if bands_idx is None:
        bands_idx = np.arange(J, dtype=np.int32)
    else:
        bands_idx = np.asarray(bands_idx, dtype=np.int32)
    B = int(bands_idx.size)

    Z_bar = (X[None, ...] + D).mean(axis=2)
    t_sec = np.arange(T, dtype=np.float64) * float(delta_spk)
    k_idx = _nearest_centres_idx(centres_sec, t_sec)

    Z_t = Z_bar[:, :, k_idx]
    phase = 2.0 * np.pi * freqs[:, None] * t_sec[None, :]
    rot = np.exp(1j * phase, dtype=np.complex128)

    Ztilt = Z_t * rot[None, :, :]
    Ztilt = np.transpose(Ztilt, (0, 2, 1))

    ZR = Ztilt.real[:, :, bands_idx].astype(np.float64)
    ZI = Ztilt.imag[:, :, bands_idx].astype(np.float64)
    return ZR, ZI, bands_idx


# ─────────────────────────────────────────────────────────────────────────────
# Config + Trace
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class JointSamplerConfig:
    n_warmup: int = 1000
    n_samples: int = 1000
    thin: int = 1
    use_pg_sampler: bool = False  # Default to False (use mean approximation)
    # WARNING: use_pg_sampler=False uses E[ω|ψ] (deterministic), NOT true MCMC sampling!
    # For exact Bayesian inference, set use_pg_sampler=True (requires polyagamma_jax).
    omega_floor: float = 1e-6

    tau2_intercept: float = 100.0**2
    tau2_beta: float = 10.0**2
    use_ard_beta: bool = False
    ard_a0_beta: float = 1e-2
    ard_b0_beta: float = 1e-2

    mu_gamma: Optional[np.ndarray] = None
    Sigma_gamma: Optional[np.ndarray] = None
    tau2_gamma: float = 25.0**2

    standardize_reim: bool = False  # WARNING: True breaks PLV/phase interpretation!
    standardize_hist: bool = False

    rng: np.random.Generator = np.random.default_rng(0)
    verbose: bool = False


@dataclass
class JointTrace:
    beta: np.ndarray
    gamma: Optional[np.ndarray]
    bands_idx: np.ndarray
    feat_mean_reim: Optional[np.ndarray]
    feat_std_reim: Optional[np.ndarray]
    feat_mean_hist: Optional[np.ndarray]
    feat_std_hist: Optional[np.ndarray]
    meta: Dict[str, Any]


# ─────────────────────────────────────────────────────────────────────────────
# JAX-optimized sampling kernels (all JIT-compiled)
# ─────────────────────────────────────────────────────────────────────────────

@jit
def compute_psi_vectorized(X: jnp.ndarray, H_all: jnp.ndarray,
                          beta: jnp.ndarray, gamma: jnp.ndarray) -> jnp.ndarray:
    """
    Compute linear predictor for all units at once.
    X: (N, p)
    H_all: (S, N, R_h)
    beta: (S, p)
    gamma: (S, R_h)
    Returns: (S, N)
    """
    psi = beta @ X.T  # (S, N)
    if gamma.shape[1] > 0:
        psi += jnp.einsum('snr,sr->sn', H_all, gamma)
    return psi


@jit
def compute_omega_mean(psi: jnp.ndarray, omega_floor: float) -> jnp.ndarray:
    """
    Compute E[ω|ψ] = 0.5 * tanh(|ψ|/2) / |ψ| (mean approximation)
    psi: (S, N)
    Returns: (S, N)
    """
    abspsi = jnp.maximum(jnp.abs(psi), 1e-12)
    omega = 0.5 * jnp.tanh(jnp.clip(abspsi, 0.0, 50.0) / 2.0) / abspsi
    return jnp.maximum(omega, omega_floor)


@jit
def sample_omega_pg(key: jax.random.PRNGKey, psi: jnp.ndarray,
                   omega_floor: float) -> jnp.ndarray:
    """
    Sample ω ~ PG(1, ψ) for all units using exact Polya-Gamma sampling.
    psi: (S, N)
    Returns: (S, N)
    """
    S, N = psi.shape
    total_samples = S * N

    # Split keys for each sample
    keys = jax.random.split(key, total_samples)

    # Flatten psi to (S*N,) for vectorized sampling
    psi_flat = psi.ravel()

    # Sample using vectorized PG sampler: ω ~ PG(1, ψ)
    omega_flat = vmap(lambda k, z: sample_pg_single(k, 1.0, z))(keys, psi_flat)

    # Reshape back to (S, N)
    omega = omega_flat.reshape(S, N)

    # Apply floor for numerical stability
    return jnp.maximum(omega, omega_floor)


@jit
def build_normal_equations_vectorized(
    X: jnp.ndarray,
    H_all: jnp.ndarray,
    omega: jnp.ndarray,
    XT_kappa: jnp.ndarray,
    HT_kappa: jnp.ndarray,
    Prec_beta_all: jnp.ndarray,
    Prec_gamma: jnp.ndarray,
    mu_gamma: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Build block normal equations for all units simultaneously.
    Returns A: (S, d, d), b: (S, d) where d = p + R_h
    """
    S, N = omega.shape
    p = X.shape[1]
    R_h = H_all.shape[2]
    d = p + R_h

    # Compute weighted X^T X more efficiently
    # A11 = X^T diag(ω) X for each unit
    # Use broadcasting: X (N,p) * sqrt(ω) (S,N,1) gives (S,N,p)
    sqrt_omega = jnp.sqrt(omega)[:, :, None]  # (S, N, 1)
    X_weighted = sqrt_omega * X[None, :, :]    # (S, N, p)
    A11 = jnp.einsum('snp,snq->spq', X_weighted, X_weighted)  # (S, p, p)

    # Add prior precision
    A11 += Prec_beta_all[:, :, None] * jnp.eye(p)[None, :, :]

    b1 = XT_kappa

    if R_h == 0:
        return A11, b1

    # A12 = X^T Ω H
    H_weighted = sqrt_omega * H_all  # (S, N, R_h)
    A12 = jnp.einsum('snp,snr->spr', X_weighted, H_weighted)

    # A22 = H^T Ω H + Prec_gamma
    A22 = jnp.einsum('snr,snk->srk', H_weighted, H_weighted)
    A22 += Prec_gamma

    # b2 with prior mean
    b2 = HT_kappa
    if mu_gamma is not None:
        b2 += jnp.einsum('srk,sk->sr', Prec_gamma, mu_gamma)

    # Assemble block matrix efficiently
    A = jnp.zeros((S, d, d))
    A = A.at[:, :p, :p].set(A11)
    A = A.at[:, :p, p:].set(A12)
    A = A.at[:, p:, :p].set(jnp.swapaxes(A12, 1, 2))
    A = A.at[:, p:, p:].set(A22)

    b = jnp.concatenate([b1, b2], axis=1)

    return A, b


@partial(jit, static_argnames=['d'])
def sample_theta_single_unit(key: jax.random.PRNGKey, A: jnp.ndarray,
                             b: jnp.ndarray, d: int) -> jnp.ndarray:
    """
    Sample θ ~ N(A^{-1}b, A^{-1}) using Cholesky decomposition.
    More robust version with better numerical stability.
    """
    # Symmetrize
    A_sym = 0.5 * (A + A.T)

    # Add small jitter for numerical stability
    A_reg = A_sym + 1e-8 * jnp.eye(d)

    # Cholesky decomposition
    L = jnp.linalg.cholesky(A_reg)

    # Solve for mean: μ = A^{-1} b
    v = jax.scipy.linalg.solve_triangular(L, b, lower=True)
    mu = jax.scipy.linalg.solve_triangular(L.T, v, lower=False)

    # Sample: θ = μ + L^{-T} ε
    eps = jax.random.normal(key, shape=(d,))
    theta = mu + jax.scipy.linalg.solve_triangular(L.T, eps, lower=False)

    return theta


# Vectorize over units
sample_theta_all_units = vmap(sample_theta_single_unit, in_axes=(0, 0, 0, None))


@jit
def update_ard_tau2(key: jax.random.PRNGKey, beta: jnp.ndarray,
                   a0: float, b0: float) -> jnp.ndarray:
    """
    Update ARD variance parameters for all units (excluding intercept).
    beta: (S, p)
    Returns: (S, p-1)
    """
    b_lat = beta[:, 1:]  # (S, p-1)
    S, p_minus_1 = b_lat.shape

    a_post = a0 + 0.5
    b_post = b0 + 0.5 * (b_lat ** 2)

    # Sample from Inverse-Gamma
    keys = jax.random.split(key, S * p_minus_1)
    keys = keys.reshape(S, p_minus_1, 2)

    def sample_inv_gamma(key_pair, b_val):
        return 1.0 / jax.random.gamma(key_pair, a_post) * b_val

    tau2 = vmap(vmap(sample_inv_gamma))(keys, b_post)

    return tau2


@partial(jit, static_argnames=['d', 'p', 'R_h', 'S', 'use_ard', 'use_pg'])
def gibbs_iteration(
    beta: jnp.ndarray,
    gamma: jnp.ndarray,
    Prec_beta_all: jnp.ndarray,
    key: jax.random.PRNGKey,
    X: jnp.ndarray,
    H_all: jnp.ndarray,
    XT_kappa: jnp.ndarray,
    HT_kappa: jnp.ndarray,
    Prec_gamma: jnp.ndarray,
    mu_gamma: jnp.ndarray,
    omega_floor: float,
    a0_ard: float,
    b0_ard: float,
    d: int,
    p: int,
    R_h: int,
    S: int,
    use_ard: bool,
    use_pg: bool,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Single Gibbs iteration (fully JIT-compiled).

    NOTE: If use_ard=True, implements PER-UNIT ARD where each unit s and feature j
    gets its own variance τ²_{s,j}. This is NOT shared across units.

    Returns: (beta_new, gamma_new, Prec_beta_all_new)
    """
    # Compute ψ
    psi = compute_psi_vectorized(X, H_all, beta, gamma)

    # Sample or approximate ω
    key_omega, key_rest = jax.random.split(key)

    def sample_omega_fn(key_psi):
        k, psi_val = key_psi
        return sample_omega_pg(k, psi_val, omega_floor)

    def approx_omega_fn(key_psi):
        _, psi_val = key_psi
        return compute_omega_mean(psi_val, omega_floor)

    omega = lax.cond(
        use_pg,
        sample_omega_fn,
        approx_omega_fn,
        (key_omega, psi)
    )

    # Build normal equations
    A, b = build_normal_equations_vectorized(
        X, H_all, omega, XT_kappa, HT_kappa,
        Prec_beta_all, Prec_gamma, mu_gamma
    )

    # Sample θ for all units (S is a static arg, no shape extraction needed)
    key_theta, key_rest = jax.random.split(key_rest)
    keys = jax.random.split(key_theta, S)
    theta = sample_theta_all_units(keys, A, b, d)

    beta_new = theta[:, :p]
    gamma_new = theta[:, p:] if R_h > 0 else gamma

    # ARD update (conditional)
    def do_ard_update(key_beta_prec):
        key_ard, beta_val, prec_val = key_beta_prec
        tau2_lat = update_ard_tau2(key_ard, beta_val, a0_ard, b0_ard)
        new_prec = prec_val.at[:, 1:].set(1.0 / jnp.maximum(tau2_lat, 1e-12))
        return new_prec

    def skip_ard_update(key_beta_prec):
        return key_beta_prec[2]

    key_ard, _ = jax.random.split(key_rest)
    Prec_beta_all_new = lax.cond(
        use_ard,
        do_ard_update,
        skip_ard_update,
        (key_ard, beta_new, Prec_beta_all)
    )

    return beta_new, gamma_new, Prec_beta_all_new


# ─────────────────────────────────────────────────────────────────────────────
# Main sampler
# ─────────────────────────────────────────────────────────────────────────────

def sample_beta_gamma_from_fixed_latents_joint(
    *,
    spikes: np.ndarray,
    H_hist: Optional[np.ndarray],
    X_mean: np.ndarray,
    D_mean: np.ndarray,
    freqs_hz: np.ndarray,
    centres_sec: np.ndarray,
    delta_spk: float,
    bands_idx: Optional[Sequence[int]] = None,
    cfg: JointSamplerConfig = JointSamplerConfig(),
) -> JointTrace:

    # Check if PG sampler is available when requested
    if cfg.use_pg_sampler and not _HAS_PG:
        raise RuntimeError(
            "use_pg_sampler=True but polyagamma_jax is not available. "
            "Either install polyagamma_jax or set use_pg_sampler=False."
        )

    if cfg.verbose:
        pg_method = "exact PG sampling" if cfg.use_pg_sampler else "mean approximation"
        print(f"[JAX sampler] Using {pg_method} for ω")
        print("[JAX sampler] Preprocessing data...")

    # ----- Preprocessing (NumPy) -----
    spikes = np.asarray(spikes, np.uint8)
    R, S, T = spikes.shape
    ZR, ZI, bidx = build_rotated_predictors_from_em(
        X_mean=X_mean, D_mean=D_mean, freqs_hz=freqs_hz,
        delta_spk=delta_spk, centres_sec=centres_sec, T=T, bands_idx=bands_idx
    )

    B = int(ZR.shape[2])
    N = R * T
    p = 1 + 2 * B

    # Design matrix X
    X = np.empty((N, p), dtype=np.float64)
    X[:, 0] = 1.0
    X[:, 1:1+B] = ZR.reshape(N, B)
    X[:, 1+B:1+2*B] = ZI.reshape(N, B)

    feat_m_reim = feat_s_reim = None
    if cfg.standardize_reim:
        feat = X[:, 1:]
        feat_m_reim = feat.mean(axis=0).astype(np.float32)
        feat_s_reim = (feat.std(axis=0) + 1e-8).astype(np.float32)
        X[:, 1:] = (feat - feat_m_reim) / feat_s_reim

    # Spike history
    if H_hist is None:
        R_h = 0
        H_all = np.zeros((S, N, 0), dtype=np.float64)
    else:
        H_hist = np.asarray(H_hist)
        if H_hist.ndim == 3:
            S2, T2, R_h = H_hist.shape
            assert S2 == S and T2 == T
            H_all = np.repeat(H_hist[:, None, :, :], R, axis=1).reshape(S, N, R_h)
        elif H_hist.ndim == 4:
            R2, S2, T2, R_h = H_hist.shape
            assert R2 == R and S2 == S and T2 == T
            H_all = np.moveaxis(H_hist, 1, 0).reshape(S, N, R_h)
        else:
            raise ValueError("H_hist must be (S,T,R_h) or (R,S,T,R_h)")

    feat_m_hist = feat_s_hist = None
    if cfg.standardize_hist and R_h > 0:
        H_stack = H_all.reshape(S*N, R_h)
        feat_m_hist = H_stack.mean(axis=0).astype(np.float32)
        feat_s_hist = (H_stack.std(axis=0) + 1e-8).astype(np.float32)
        H_all = (H_all - feat_m_hist[None, None, :]) / feat_s_hist[None, None, :]

    # Responses
    Y_all = spikes.transpose(1, 0, 2).reshape(S, N).astype(np.float64)
    kappa = Y_all - 0.5

    if cfg.verbose:
        print(f"[JAX sampler] Data shapes: S={S}, N={N}, p={p}, R_h={R_h}")
        print("[JAX sampler] Converting to JAX arrays...")

    # ----- Convert to JAX arrays -----
    X_jax = jnp.array(X)
    H_all_jax = jnp.array(H_all)
    kappa_jax = jnp.array(kappa)

    # Precompute X^T κ and H^T κ
    XT_kappa = jnp.einsum('np,sn->sp', X_jax, kappa_jax)
    HT_kappa = jnp.zeros((S, R_h)) if R_h == 0 else jnp.einsum('snr,sn->sr', H_all_jax, kappa_jax)

    # ----- Priors -----
    Prec_beta_base = np.zeros(p, dtype=np.float64)
    Prec_beta_base[0] = 1.0 / max(cfg.tau2_intercept, 1e-12)
    Prec_beta_base[1:] = 1.0 / max(cfg.tau2_beta, 1e-12)

    # Initialize precision matrix (same for ARD and non-ARD; ARD updates happen in gibbs_iteration)
    Prec_beta_all = jnp.broadcast_to(Prec_beta_base, (S, p))

    # Gamma prior
    if R_h > 0 and cfg.Sigma_gamma is not None:
        Sg = np.asarray(cfg.Sigma_gamma, float)
        if Sg.ndim == 2:
            Prec_gamma = np.linalg.pinv(Sg)
            Prec_gamma = np.broadcast_to(Prec_gamma, (S, R_h, R_h)).copy()
        else:
            Prec_gamma = np.stack([np.linalg.pinv(Sg[s]) for s in range(S)], axis=0)
    else:
        var = 0.0 if R_h == 0 else 1.0 / max(cfg.tau2_gamma, 1e-12)
        Prec_gamma = np.broadcast_to(np.eye(max(R_h, 1)) * var, (S, max(R_h, 1), max(R_h, 1))).copy()
        if R_h == 0:
            Prec_gamma = Prec_gamma[:, :0, :0]
    Prec_gamma = jnp.array(Prec_gamma)

    mu_gamma_jax = None
    if R_h > 0 and cfg.mu_gamma is not None:
        mu_gamma_jax = jnp.broadcast_to(jnp.array(cfg.mu_gamma).reshape(1, R_h), (S, R_h))

    # ----- Initialize -----
    pbar = Y_all.mean(axis=1)
    ok = (pbar > 0.0) & (pbar < 1.0)
    beta_init = jnp.zeros((S, p))
    beta_init = beta_init.at[ok, 0].set(jnp.log(pbar[ok] / (1.0 - pbar[ok])))
    beta = beta_init
    gamma = jnp.zeros((S, max(R_h, 1)))
    if R_h == 0:
        gamma = gamma[:, :0]

    if cfg.verbose:
        print("[JAX sampler] Starting Gibbs sampling...")
        print(f"[JAX sampler] Warmup: {cfg.n_warmup}, Samples: {cfg.n_samples}, Thin: {cfg.thin}")

    # ----- Sampling loop with progress monitoring -----
    total = int(cfg.n_warmup + cfg.n_samples * cfg.thin)
    d = p + R_h

    key = jax.random.PRNGKey(cfg.rng.integers(0, 2**31))

    # Partial application for fixed arguments (compiles once on first call)
    gibbs_fn = partial(
        gibbs_iteration,
        X=X_jax,
        H_all=H_all_jax,
        XT_kappa=XT_kappa,
        HT_kappa=HT_kappa,
        Prec_gamma=Prec_gamma,
        mu_gamma=mu_gamma_jax,
        omega_floor=cfg.omega_floor,
        a0_ard=cfg.ard_a0_beta,
        b0_ard=cfg.ard_b0_beta,
        d=d,
        p=p,
        R_h=R_h,
        S=S,
        use_ard=cfg.use_ard_beta,
        use_pg=cfg.use_pg_sampler,
    )

    # Pre-allocate storage for posterior samples
    beta_draws = []
    gamma_draws = [] if R_h > 0 else None

    # Run iterations with progress monitoring
    for i in range(total):
        key, subkey = jax.random.split(key)

        # Progress monitoring every 10 iterations
        if cfg.verbose and (i + 1) % 10 == 0:
            phase = "Warmup" if i < cfg.n_warmup else "Sampling"
            iter_in_phase = i + 1 if i < cfg.n_warmup else i + 1 - cfg.n_warmup
            print(f"[JAX sampler] {phase} iteration {iter_in_phase}/{cfg.n_warmup if i < cfg.n_warmup else cfg.n_samples * cfg.thin}")

        # Single Gibbs iteration (JIT-compiled, reuses compilation after first call)
        beta, gamma, Prec_beta_all = gibbs_fn(beta, gamma, Prec_beta_all, subkey)

        # Store samples after warmup, with thinning
        if i >= cfg.n_warmup and (i - cfg.n_warmup) % cfg.thin == 0:
            beta_draws.append(np.array(beta, dtype=np.float32))
            if R_h > 0:
                gamma_draws.append(np.array(gamma, dtype=np.float32))

    # Convert lists to arrays
    beta_draws = np.array(beta_draws)
    gamma_draws = None if R_h == 0 else np.array(gamma_draws)

    if cfg.verbose:
        print(f"[JAX sampler] Returned {beta_draws.shape[0]} posterior samples")

    meta = dict(
        n_warmup=cfg.n_warmup,
        n_samples=cfg.n_samples,
        thin=cfg.thin,
        use_pg_sampler=cfg.use_pg_sampler,
        B=B, p=p, R_h=R_h
    )

    return JointTrace(
        beta=beta_draws,
        gamma=gamma_draws,
        bands_idx=bidx,
        feat_mean_reim=feat_m_reim,
        feat_std_reim=feat_s_reim,
        feat_mean_hist=feat_m_hist,
        feat_std_hist=feat_s_hist,
        meta=meta
    )
