# Performance Comparison: JAX vs C-backend

This document summarizes the performance comparison between the JAX implementation and the C-backend for Polya-Gamma sampling.

## Running the Comparison

```bash
# Basic usage with defaults (h=1.0, n=10000)
python compare_polyagamma_methods.py

# Custom parameters
python compare_polyagamma_methods.py --h 1.0 --z-values 0.1 0.5 2.0 10.0 --n-samples 10000
```

## Key Findings

### Summary (h=1.0, n=10000 samples)

| Method   | z=0.1  | z=0.5  | z=2.0  | z=10.0 | Notes |
|----------|--------|--------|--------|--------|-------|
| Devroye  | 0.03x  | 0.04x  | 0.36x  | 2.90x  | Faster for large z |
| Saddle   | 1.15x  | 1.05x  | 1.30x  | 4.21x  | Competitive across all z |
| Normal   | 140x   | 170x   | 180x   | 187x   | Dramatically faster |
| Hybrid   | 0.03x  | 0.04x  | 0.39x  | 2.54x  | Auto-selects best method |

*Values show speedup factor (JAX time / C time); higher is better*

### Analysis

1. **Normal Approximation**: Shows exceptional performance (140x-187x speedup) due to:
   - Simple mathematical operations that vectorize well
   - No complex rejection sampling loops
   - Ideal for GPU/parallel execution

2. **Saddle Point Method**: Competitive with C-backend (1.05x-4.21x):
   - Better performance for larger z values
   - Good balance between accuracy and speed
   - Scales well with batch size

3. **Devroye Method**: Performance varies with z:
   - Slower for small z (0.03x-0.04x) due to complex rejection sampling
   - Faster for large z (2.90x at z=10.0)
   - Best for exact sampling when h is integer

4. **Hybrid Method**: Automatically selects the best algorithm:
   - Matches Devroye for h=1, small z
   - Matches Saddle for intermediate cases
   - Matches Normal for large h (h>50)

## Implementation Details

The comparison script uses:
- **Batch sampling**: Creates arrays of identical z values for vectorized operations
- **JIT compilation warmup**: 3 warmup runs to ensure code is compiled
- **Best-of-5 timing**: Reports minimum time from 5 runs for consistent results
- **Fair comparison**: Same statistical samples, proper synchronization

## When to Use Each Method

- **Normal**: When h > 50 or when maximum speed is required and approximation is acceptable
- **Saddle**: General-purpose method, good for h ≥ 8 or (h > 4 and z ≤ 4)
- **Devroye**: When exact sampling is needed for small integer h
- **Hybrid**: Let the library choose automatically based on (h, z)

## Batch Size Considerations

JAX performance improves significantly with larger batch sizes:
- GPU acceleration benefits larger batches
- Overhead of JIT compilation amortizes over more samples
- Vectorization efficiency increases

For production use, consider batching operations when possible to maximize JAX performance.
