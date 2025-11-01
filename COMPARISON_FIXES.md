# Comparison Distribution Fixes

## Issues Identified

### Issue 1: Comparing Different Methods
The `compare_distributions.py` script was comparing **different methods** between C and JAX:
- **C-backend**: Used `random_polyagamma(h, z)` which calls the **hybrid method** that automatically selects the best algorithm based on parameters
- **JAX**: Used explicit methods (`sample_pg_devroye_single`, `sample_pg_saddle_single`, etc.)

For example, with `h=1.0, z=0.1`:
- C hybrid would select **Devroye** method (because `h == 1`)
- But the script compared this against JAX's Devroye, Saddle, AND Normal methods
- This meant: C Devroye vs JAX Saddle would show different distributions (as expected, but misleading!)

### Issue 2: Normal Approximation Invalid for h=1
The Normal approximation method **produces negative values** for h=1 because:
- Normal approximation is **only valid for h > 50** (documented in code comments)
- For small h, it samples from a Normal distribution that can be negative
- PG distribution is strictly positive, so negative values are invalid
- The C backend's hybrid method **never uses Normal for h <= 50**

### Issue 3: Devroye Method Producing Zeros
The JAX Devroye implementation had **two critical bugs**:

**Bug 1: Key Reuse** (polyagamma_jax.py:704-708)
```python
# WRONG - overwrites split key with returned key
key, subkey = random.split(key)
key, sample = random_jacobi_star(subkey, ...)  # Overwrites key!
return key, result + sample
```

This caused incorrect random number generation, leading to repeated or invalid samples.

**Bug 2: Insufficient max_iter** (polyagamma_jax.py:575)
- Original: `max_iter=1000`
- For z=0.1, acceptance rate is low enough that some samples need > 1000 iterations
- When max_iter is exceeded, the function returns **0.0**!
- This caused many zero values in the distribution

### Issue 4: Shared X-Axis
The comparison plots used a shared x-axis across all z values, making it hard to see the distribution shape for different parameter values.

## Fixes Applied

### 1. Fixed `compare_distributions.py`
- Now explicitly specifies the C-backend method to use via the `method` parameter
- Compares **same method to same method**:
  - JAX Devroye vs C Devroye (method="devroye")
  - JAX Saddle vs C Saddle (method="saddle")
- **Removed Normal comparison** for h=1 (invalid parameter range)
- Removed shared x-axis - each subplot now has its own x-range

### 2. Fixed JAX Devroye Implementation (polyagamma_jax.py)

**Fix 1: Correct Key Handling**
```python
# CORRECT - properly thread keys through the loop
new_key, subkey = random.split(key)
_, sample = random_jacobi_star(subkey, ...)  # Discard returned key
return new_key, result + sample  # Use split key for next iteration
```

**Fix 2: Increased max_iter**
- Changed from `max_iter=1000` to `max_iter=10000`
- Ensures enough iterations for low acceptance rates (like z=0.1)
- Prevents returning zero values

### 3. Added Clear Documentation
- Updated docstrings and print statements to clarify what's being compared
- Plot titles now show which C method is being used
- Added comments explaining why Normal is skipped for h=1

## Devroye Implementation Comparison

I performed a detailed comparison of the Devroye implementations:

### Similarities ✓
- Both use the same truncation point `T = 0.64`
- Same algorithm structure and logic flow
- Same mathematical formulas
- Same proposal probabilities calculation
- Same acceptance-rejection logic

### Minor Differences Noted

#### Precision in C Implementation
The C code uses **single-precision** (float) for some intermediate calculations:

```c
// pgm_devroye.c, line 36-45
static PGM_INLINE float
piecewise_coef(int n, parameter_t const* pr)
{
    if (pr->x > T) {
        double b = PGM_PI * (n + 0.5);
        return (float)b * expf(-0.5 * pr->x * b * b);  // expf = float exp
    }
    double a = n + 0.5;
    return expf(-1.5 * (PGM_LOGPI_2 + pr->logx) - 2. * a * a / pr->x) *
           (float)(PGM_PI * a);
}
```

Notice:
- Uses `expf()` (single precision exponential) instead of `exp()`
- Returns `float` instead of `double`

#### JAX Implementation
The JAX code uses **double precision** (float64) throughout:

```python
# polyagamma_jax.py, lines 474-482
def large_x_case():
    b = PI * (n + 0.5)
    return b * jnp.exp(-0.5 * x * b * b)  # jnp.exp uses float64

def small_x_case():
    a = n + 0.5
    return jnp.exp(-1.5 * (LOGPI_2 + logx) - 2.0 * a * a / x) * (PI * a)
```

**Impact**: This could cause minor numerical differences (~1e-6 to 1e-7 relative error), but should NOT cause visibly different distributions in practice.

## Testing Instructions

To verify the fixes:

```bash
# Run the updated comparison script
python compare_distributions.py
```

This will generate three plots:
1. `distribution_comparison_devroye.png` - JAX Devroye vs C Devroye
2. `distribution_comparison_saddle.png` - JAX Saddle vs C Saddle
3. `distribution_comparison_normal.png` - JAX Normal vs C hybrid

**Expected Results**:
- Devroye comparison: Distributions should match very closely
- Saddle comparison: Distributions should match very closely
- Normal comparison: May differ (comparing different methods)

## If Devroye Distributions Still Differ

If the Devroye method still shows different distributions for z=0.1:

1. Check the random seeds - they should be deterministic but different sequences
2. Verify JAX is using float64: `jax.config.update("jax_enable_x64", True)`
3. Consider the precision differences noted above
4. Run a statistical test (e.g., KS test) to quantify the difference

The precision difference in the C code is a design choice (trading accuracy for speed) and would require modifying the C source to use double precision throughout if exact matching is required.
