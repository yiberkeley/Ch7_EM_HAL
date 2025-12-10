# HAL-EM Documentation: Comprehensive Reference

## Overview

Implementation of **Highly Adaptive Lasso (HAL)** for censored data using **EM algorithm**. Supports:
- **Interval Censoring (IC)**: T ∈ [L, R]
- **Right Censoring (RC)**: min(T, C) with δ = 𝟙{T ≤ C}

---

## Mathematical Framework

### HAL Basis Functions

**Zero-Order** (smoothness=0): φₖ(x) = 𝟙{x ≥ gₖ} - Step functions
**First-Order** (smoothness=1): φₖ(x) = (x - gₖ)₊ - Piecewise linear

**Log-Density**: log f(x; θ) = θ₀ + Σₖ θₖ φₖ(x)

**Regularization**: minimize -ℓ(θ) + λ||θ||₁ (L1 penalty for sparsity)

---

## Key Hyperparameters

### 1. Knot Selection
- **Formula**: K = round(10√n)
- **Examples**: n=200→141 knots, n=800→283 knots, n=3200→566 knots
- **Placement**: Percentile-based on data

### 2. Regularization (λ)
- **First-order HAL**: λ ∈ [60, 65, 70, 80]
- **Zero-order HAL**: λ ∈ [3, 5, 8, 10]
- **Selection**: 5-fold cross-validation (max validation log-likelihood)
- **EM iterations**: 5×λ_CV (relaxed constraint)

### 3. Pruning & Convergence
- **Threshold**: 1e-4 (removes near-zero coefficients)
- **Max iterations**: 100
- **Tolerance**: 0.05 (absolute ΔLL)
- **Imputations**: 200 samples per censored observation

### 4. Integration Grids

| Grid Type | IC | RC | Purpose |
|-----------|----|----|---------|
| **HAL Fitting** | 200 pts [0,1] | 200 pts [0,1] | Density optimization |
| **Survival Integration** | 1000 pts [0,1] | 1000 pts [0,1] | Accurate CDF computation |
| **Evaluation Output** | 100 pts [0,1] | 20 pts [0.02,0.98] | CSV output |
| **Analysis (filtered)** | 52 pts [0.242,0.758] | 10 pts [0.242,0.758] | Performance metrics |

**Key**: Survival uses **1000-point grid** for integration accuracy, NOT the 200-point fitting grid.

---

## Implementation Details

### Initial Imputation Strategies
- **IC Midpoint**: W₁ = (L+R)/2 (fast, biased for skewed densities)
- **IC Uniform**: T ~ Unif(L, R) (unbiased, slower convergence)

### IPCW Weighting (RC)
- **Weights**: wᵢ = 1/Ĝ(T̃ᵢ) for uncensored observations
- **Ĝ(t)**: Kaplan-Meier estimate of P(C ≥ t)
- **Purpose**: Correct for informative dropout under MAR

### Numerical Integration
- **Riemann Sum**: O(Δx²) accuracy, used for HAL fitting
- **Trapezoidal Rule**: O(Δx³) accuracy, used for survival computation
- **Exact (First-order)**: ∫ exp(a+bx)dx = exp(a)/b · expm1(b·Δx)

---

## Algorithm Flow

```
1. INITIALIZATION
   - Impute: W₁⁽⁰⁾ ← midpoint/uniform
   - Select knots: K = 10√n (percentile-based)
   - Solve HAL: θ⁽⁰⁾ with λ from CV
   - Prune: Keep |θₖ| > 1e-4

2. EM ITERATIONS (until |ΔLL| < 0.05 or max 100 iters)

   E-step:
   - Compute F(x|θ⁽ᵗ⁻¹⁾) on 500-point CDF grid
   - IC: Sample M=200 points from truncated [Lᵢ, Rᵢ]
   - RC: Sample M=200 points from truncated [T̃ᵢ, ∞)

   M-step:
   - Solve HAL with augmented data: θ⁽ᵗ⁾
   - Use 5×λ_CV constraint (relaxed)
   - Warm start from θ⁽ᵗ⁻¹⁾
   - Normalize: ∫f(x)dx = 1

3. INFERENCE
   - Compute scores: sᵢ = E[φ(X)|data] - E[φ(X)]
   - Fisher info: Î = n⁻¹Σsᵢsᵢᵀ
   - Covariance: Cov(β̂) = (Î + ridge·I)⁻¹/n
   - Ridge: 1e-6/√p (stability, minimal bias)

4. SURVIVAL COMPUTATION
   - Evaluate density on 1000-point grid
   - Integrate: Ŝ(x) = ∫ₓ¹ f̂(t)dt
   - Interpolate to evaluation grid (100 or 20 points)
```

---

## Numerical Stability

### Log-Sum-Exp Trick
```python
max_val = np.max(log_density)
log_Z = max_val + np.log(np.sum(np.exp(log_density - max_val) * delta))
```
Prevents overflow/underflow when computing normalization constant.

### Solver Configuration
- **Primary: ECOS** - Fast, high accuracy (ε≈10⁻⁸)
- **Fallback: SCS** - Robust to ill-conditioning (ε≈10⁻⁴)
- **Warm starts**: 5-10× speedup in EM iterations

### Normalization Strategies
- **Post-integration**: Force Σfⱼ·Δⱼ = 1 exactly
- **CDF endpoint**: Force F(1) = 1.0 (prevents S(1)<0)

---

## Hyperparameter Verification (IC vs RC)

**Do table values apply to both IC and RC?** → **PARTIALLY YES**

| Parameter | Table Value | IC | RC | Match? |
|-----------|-------------|----|----|---------|
| Basis order | First-order (s=1) | ✅ | ✅ | YES |
| Knots | K = 10√n | ✅ | ✅ | YES |
| Pruning | ε = 1e-4 | ✅ | ✅ | YES |
| EM constraint | 5×λ_CV | ✅ | ✅ | YES |
| E-step samples | 200 | ✅ | ✅ | YES |
| Max iterations | 100 | ✅ | ✅ | YES |
| Tolerance | 0.05 | ✅ | ✅ | YES |
| **Evaluation grid** | **1000 pts [0,1]** | **100 pts** | **20 pts** | **NO** |

**Clarification**: 1000-point grid refers to **internal survival integration**, not output evaluation grid.

---

## Grid Usage Summary

**Four distinct grids** are used in the pipeline:

1. **HAL Fitting Grid** (200 pts): Internal optimization
2. **Survival Integration Grid** (1000 pts): Accurate CDF computation ⭐
3. **Evaluation Grid** (IC: 100, RC: 20): Output saved to CSV
4. **Analysis Grid** (filtered subset): Inner 95% for performance metrics

**Critical**: Survival does NOT use 200-point fitting grid - it uses 1000 points for integration accuracy, then interpolates to evaluation grid.

**Computational Flow**:
```
HAL fit (200 pts) → Survival integration (1000 pts) →
Interpolate to eval grid (100/20 pts) → Save CSV → Analysis (full or filtered)
```

---

## Performance Benchmarks

| n | K | λ (CV) | EM iters | Total time | Final LL |
|---|---|--------|----------|------------|----------|
| 200 | 141 | 65 | 12 | 3 min | -95 |
| 800 | 283 | 65 | 8 | 15 min | -430 |
| 3200 | 566 | 70 | 6 | 80 min | -1740 |

**Complexity**: Time ≈ O(n^(3/2)) due to K~√n and EM iterations~log(n)

---

## Common Issues & Solutions

### EM Divergence (LL decreases)
- **Causes**: Overflow, coarse grid, λ too small
- **Solutions**: Check log_density < 100, increase grid to 500 pts, increase λ to 10×

### Singular Fisher Information
- **Causes**: Collinear basis, too many knots, all coefficients pruned
- **Solutions**: Increase ridge to 1e-4, use pseudoinverse, reduce K to 5√n

### Extreme IPCW Weights (RC)
- **Causes**: Heavy late censoring, G(t)→0
- **Solutions**: Winsorize at 99th percentile, administrative censoring at τ < max(T)

---

## Key Differences: IC vs RC

| Aspect | IC | RC |
|--------|----|----|
| E-step method | Sample from [Lᵢ, Rᵢ] | Sample from [T̃ᵢ, ∞) |
| E-step samples | 200 per interval | 200 per censored obs |
| Weighting | Uniform | IPCW (1/Ĝ(T)) |
| Output grid | 100 pts [0,1] | 20 pts [0.02,0.98] |
| Grid range | Full support | Trimmed (avoid boundaries) |

**Both use identical**:
- HAL fitting grid (200 pts)
- Survival integration grid (1000 pts)
- EM convergence criteria
- Regularization strategy

---

## References

1. **van der Laan, M. J.** (2017). HAL-based Targeted Minimum Loss Estimator. *Int J Biostatistics*.
2. **van der Laan & Bibaut** (2017). Uniform Consistency of HAL. *arXiv:1709.06256*.
3. **Dempster, Laird & Rubin** (1977). EM Algorithm. *JRSS-B*.
4. **Robins & Rotnitzky** (1992). IPCW for Dependent Censoring. *AIDS Epidemiology*.

---

**Version**: 1.0 | **Last Updated**: 2024
**Codebase**: [ic_draft.py](Asymptoticity_IC_0301/ic_draft.py), [rc_draft.py](Asymptoticity_RC_0301/rc_draft.py)
