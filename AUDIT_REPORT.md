# Audit Report: beta_gamma_pg_fixed_latents_joint_jax.py

## Executive Summary

**Overall Status**: ✅ Implementation is algorithmically correct with minor code quality issues.

**Compilation**: ✅ Will compile once per configuration (mostly safe, with caveats below).

**Algorithm**: ✅ PG-augmented Gibbs sampler is mathematically correct for fixed latents.

---

## 1. COMPILE-ONCE ANALYSIS

### ✅ PASS: Static Shapes
- **Lines 401-410**: `S, N, p, R_h, d` computed as concrete Python ints from data shapes
- All bound in `partial()` at line 515-531 as static args `d`, `p`, `R_h`
- **Verdict**: Shapes are fixed within a run → **no recompilation from shape changes**

### ✅ PASS: Static Flags
- **Line 288**: `use_pg` and `use_ard` marked as `static_argnames` in `@partial(jit, ...)`
- Bound once in `partial()` at lines 529-530
- **Verdict**: Flags are static → **no recompilation when toggling within a single sampler instance**

### ✅ PASS: mu_gamma None-ness is Fixed
- **Lines 490-492**: `mu_gamma_jax` set to `None` or array before loop
- Bound in `partial()` at line 522, never changes
- **Verdict**: None-ness is fixed → **no recompilation from conditional mu_gamma**

### ⚠️ MINOR CONCERN: Shape Extraction from Traced Arrays

**Issue 1: Line 341 in `gibbs_iteration`**
```python
S = beta.shape[0]  # Extracting from traced array
key_theta, key_rest = jax.random.split(key_rest)
keys = jax.random.split(key_theta, S)  # Using extracted S
```

**Analysis**:
- `beta` is a traced array with static shape `(S, p)`
- In JAX, `beta.shape` returns concrete ints when shape is static
- `random.split(key, S)` requires concrete `S` (not traced)
- **Current behavior**: Should work because shapes are static
- **Risk**: If shapes ever become dynamic, this will fail with ConcretizationTypeError

**Recommendation**: Add `S` to `static_argnames` and pass explicitly:
```python
@partial(jit, static_argnames=['d', 'p', 'R_h', 'S', 'use_ard', 'use_pg'])
def gibbs_iteration(..., S: int, ...):
    ...
    keys = jax.random.split(key_theta, S)  # Use static S
```

**Issue 2: Line 152 in `sample_omega_pg`**
```python
S, N = psi.shape
total_samples = S * N
keys = jax.random.split(key, total_samples)
```

Same issue as above. **Recommendation**: Pass `S` and `N` as static args or use `psi.size` (which is always concrete for static shapes).

### ✅ PASS: Argument dtypes Don't Change
- **Lines 457-459**: All JAX arrays created with explicit `dtype=float64`
- **Lines 412, 427, 449**: NumPy arrays created with `dtype=float64` before conversion
- **Verdict**: dtypes are consistent → **no dtype-triggered recompilation**

### 📊 COMPILE-ONCE VERDICT
**Status**: ✅ **PASS with minor improvements recommended**

Will compile once per `(use_pg_sampler, use_ard_beta)` configuration. The shape extraction issues are safe with current static shapes but could be made more robust.

---

## 2. ALGORITHMIC CORRECTNESS ANALYSIS

### ✅ PASS: Design and Sufficient Statistics

#### Predictor Construction (Lines 34-68)
```python
Z_bar = (X[None, ...] + D).mean(axis=2)  # Taper average ✅
Z_t = Z_bar[:, :, k_idx]                 # Nearest center ✅
Ztilt = Z_t * exp(i 2π f t)              # Phase rotation ✅
ZR, ZI = Ztilt.real, Ztilt.imag          # Real/imag split ✅
```
**Verdict**: ✅ Correct fixed-latent predictors from EM output

#### Design Matrix (Lines 412-415)
```python
X[:, 0] = 1.0              # Intercept ✅
X[:, 1:1+B] = ZR           # Real parts ✅
X[:, 1+B:1+2*B] = ZI       # Imag parts ✅
```
**Shape**: `(N, p)` where `N = R*T`, `p = 1 + 2*B` ✅

#### Response (Lines 449-450)
```python
Y_all = spikes.transpose(1,0,2).reshape(S, N)  # ✅ Shape (S, N)
kappa = Y_all - 0.5                            # ✅ PG augmentation offset
```

#### History Handling (Lines 425-439)
- Supports `(S, T, R_h)` and `(R, S, T, R_h)` formats ✅
- Reshapes to `(S, N, R_h)` correctly ✅

### ✅ PASS: PG-Augmented Normal Equations

#### Linear Predictor (Lines 126-129)
```python
psi = beta @ X.T                             # (S,p) @ (p,N) = (S,N) ✅
psi += einsum('snr,sr->sn', H_all, gamma)    # Add history term ✅
```

#### Omega Sampling/Approximation (Lines 133-168)
- **Mean**: `E[ω|ψ] = 0.5 * tanh(|ψ|/2) / |ψ|` ✅ Correct formula
- **Exact**: `ω ~ PG(1, ψ)` via `sample_pg_single` ✅ Correct

#### Block Precision Matrix (Lines 172-228)

**A11 (Line 196)**:
```python
A11 = X^T Ω X  # via sqrt(ω) weighted einsum ✅
A11 += diag(Prec_beta)  # ✅ Diagonal prior precision
```

**A12 (Line 208)**: `X^T Ω H` ✅

**A22 (Lines 211-212)**:
```python
A22 = H^T Ω H  ✅
A22 += Prec_gamma  ✅
```

**RHS (Lines 201, 215-217)**:
```python
b1 = X^T κ  ✅
b2 = H^T κ + Prec_gamma @ mu_gamma  ✅ (if mu_gamma provided)
```

**Block Assembly (Lines 220-226)**:
```python
A = [A11  A12]  ✅
    [A21  A22]
b = [b1, b2]    ✅
```

### ✅ PASS: β/γ Blocked Draw (Lines 231-255)

```python
A_sym = 0.5 * (A + A.T)                    # Symmetrize ✅
A_reg = A_sym + 1e-8 * I                   # Jitter for stability ✅
L L^T = A_reg                              # Cholesky ✅
μ = A^{-1} b                               # via 2 triangular solves ✅
θ = μ + L^{-T} ε where ε ~ N(0,I)         # Sample ✅
```

**Verdict**: ✅ Correct Gaussian sampler with numerical stability

### ⚠️ DESIGN QUESTION: ARD Implementation (Lines 262-285)

#### Current Behavior: Per-Unit, Per-Feature ARD

```python
# Lines 270-271: Operates on each unit separately
b_lat = beta[:, 1:]  # (S, p-1) - all units, excluding intercept

# Lines 277-283: Samples τ²_{s,j} for EACH unit s, feature j
tau²_{s,j} ~ InvGamma(a0 + 0.5, b0 + 0.5 * β_{s,j}²)
```

**This means**:
- Unit 1, Feature 1 has its own variance τ²_{1,1}
- Unit 2, Feature 1 has its own variance τ²_{2,1}
- These are INDEPENDENT across units

**Alternative Design** (more common):
- Feature 1 has shared variance τ²_1 across all units
- Feature 2 has shared variance τ²_2 across all units
- Variance is feature-wise, not unit-specific

**Question for User**: Is per-unit ARD the intended design, or should variance be shared across units for each feature?

**Mathematical Correctness**: Current implementation is correct for per-unit ARD. The IG update is:
```
a_post = a0 + 0.5 ✅
b_post = b0 + 0.5 * β² ✅
τ² = b_post / Gamma(a_post, 1) ✅ (equivalent to InvGamma draw)
```

### 🐛 CODE QUALITY ISSUE: Redundant ARD Initialization (Lines 470-473)

```python
if cfg.use_ard_beta:
    Prec_beta_all = jnp.broadcast_to(Prec_beta_base, (S, p))
else:
    Prec_beta_all = jnp.broadcast_to(Prec_beta_base, (S, p))
```

**Issue**: Both branches are identical!

**Root Cause**: ARD logic is controlled by `lax.cond(use_ard, ...)` in `gibbs_iteration` (lines 360-365), not by initialization.

**Recommendation**: Simplify to:
```python
Prec_beta_all = jnp.broadcast_to(Prec_beta_base, (S, p))
```

**Impact**: No correctness issue, just dead code.

---

## 3. NUMERICAL STABILITY

### ✅ PASS: All Guards in Place

- **Line 242**: Symmetrization before Cholesky ✅
- **Line 242**: `1e-8 * I` jitter for ill-conditioned matrices ✅
- **Line 168**: `omega_floor` prevents zero weights ✅
- **Line 353**: `max(tau2, 1e-12)` prevents division by zero ✅
- **Line 139**: `max(|ψ|, 1e-12)` prevents division by zero ✅
- **Line 140**: `clip(|ψ|, 0, 50)` prevents overflow in tanh ✅

---

## 4. WHAT THIS DOES (AND DOESN'T) DO

### ✅ What It Does
1. **Correct PG-Gibbs for fixed latents**: Properly samples (β, γ) | ω, Y with PG augmentation
2. **Vectorized over units**: All S units updated in parallel (efficient)
3. **Flexible priors**: Supports both scalar and matrix priors for γ
4. **Numerically stable**: Guards against common pitfalls

### ⚠️ What It Doesn't Do (by Design)
1. **Not true sampling when `use_pg_sampler=False`**: Uses ω = E[ω|ψ] (deterministic), making this a mode-finding algorithm, not MCMC
2. **No cross-unit shrinkage**: Each unit's β is independent (unless you want to add hierarchical priors later)
3. **Standardization breaks interpretation**: If `standardize_reim=True`, βR/βI are no longer in PLV/phase units

---

## 5. RECOMMENDATIONS

### Critical: None

### Important
1. **Clarify ARD design**: Confirm per-unit ARD is intended (vs. shared feature-wise variance)

### Code Quality
1. **Remove dead code** (lines 470-473): Both branches identical
2. **Add `S` to static args** (lines 288, 144): Make shape extraction explicit
3. **Document per-unit ARD**: Add comment explaining variance is not shared across units

### Documentation
1. Add docstring warning that `use_pg_sampler=False` is not true MCMC
2. Document that `standardize_reim=True` changes interpretation of coefficients

---

## 6. FINAL VERDICT

| Criterion | Status | Notes |
|-----------|--------|-------|
| **Compile-once** | ✅ PASS | Minor shape-extraction concerns (non-critical) |
| **Algorithm** | ✅ PASS | Mathematically correct PG-Gibbs |
| **Numerics** | ✅ PASS | All stability guards in place |
| **Code Quality** | ⚠️ MINOR | Dead code in ARD initialization |

**Recommendation**: ✅ **Safe to use** with awareness of ARD design choice.

---

## Appendix: Test Checklist

To verify compile-once behavior, run:

```python
# Should compile on first call, reuse on second
trace1 = sample_beta_gamma_from_fixed_latents_joint(..., cfg=cfg)  # Compiles
trace2 = sample_beta_gamma_from_fixed_latents_joint(..., cfg=cfg)  # Reuses

# Changing PG mode triggers new compile (expected)
cfg.use_pg_sampler = True
trace3 = sample_beta_gamma_from_fixed_latents_joint(..., cfg=cfg)  # New compile

# Same PG mode reuses
trace4 = sample_beta_gamma_from_fixed_latents_joint(..., cfg=cfg)  # Reuses
```

Monitor with: `JAX_LOG_COMPILES=1 python your_script.py`
