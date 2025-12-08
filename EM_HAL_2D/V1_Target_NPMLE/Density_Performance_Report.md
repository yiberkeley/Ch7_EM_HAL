# Technical Report: Density Estimation Performance Evaluation

## Executive Summary

This report evaluates the performance of three bivariate density estimation methods using censored data: **Initial_1** (initial model after 1 EM iteration), **Final_5** (model after 5 EM iterations), and **NPMLE_Targeted** (NPMLE with targeted learning). The evaluation is based on 40 simulation runs with n=500 samples each, using a bivariate Gaussian true density with mean [0.5, 0.5] and covariance [[0.05, 0.0], [0.0, 0.05]].

### Key Findings

1. **Final_5** achieves the best overall performance with the lowest MSE (0.0825) and excellent coverage (92.5%)
2. **NPMLE_Targeted** shows comparable bias to Final_5 but with substantially higher variance (std=0.5353)
3. **Initial_1** exhibits severe positive bias (0.3576) and poor coverage (35.0%)
4. Both Final_5 and NPMLE_Targeted maintain good performance in tail regions, while Initial_1 degrades significantly

---

## 1. Methodology

### 1.1 Data Generation Procedure

The simulation data is generated using the following procedure (implemented in `censored_NPMLE_EM.py`):

#### True Density Model
The true underlying distribution is a **truncated bivariate Gaussian** distribution:
- **Mean vector**: μ = [0.5, 0.5]
- **Covariance matrix**: Σ = [[0.05, 0.0], [0.0, 0.05]] (diagonal, independent components)
- **Support**: Truncated to the unit square [0, 1] × [0, 1]
- **Sampling method**: Rejection sampling from the bivariate normal distribution

The truncation ensures all samples fall within [0, 1]² by rejecting any candidates outside this region.

#### Censoring Mechanism
For each simulated dataset:

1. **Generate complete event times**: Draw n = 500 samples (T1, T2) from the truncated bivariate Gaussian
2. **Generate censoring times**:
   - C1 ~ Uniform(0, 1) for dimension 1
   - C2 ~ Uniform(0, 1) for dimension 2
   - Censoring times are independent of event times
3. **Apply right-censoring**:
   - Observed time: T̃1 = min(T1, C1), T̃2 = min(T2, C2)
   - Event indicators: δ1 = I(T1 ≤ C1), δ2 = I(T2 ≤ C2)

The final observed data consists of (T̃1, δ1, T̃2, δ2) for each observation, with approximately 50% censoring rate due to the uniform censoring mechanism.

#### Random Seed
Each simulation run uses a unique random seed: `4250 + sim`, where `sim` is the simulation index (0, 1, ..., 39), ensuring reproducibility while creating different datasets across simulations.

### 1.2 Simulation Setup
- **Sample size**: n = 500 per simulation
- **Number of simulations**: 40
- **True density**: Truncated bivariate Gaussian with μ = [0.5, 0.5] and Σ = [[0.05, 0.0], [0.0, 0.05]]
- **Censoring**: Independent uniform censoring on [0, 1] for each dimension (~50% censoring rate)
- **Evaluation grid**: 100 grid points (10×10 grid over [0, 1]²)
- **Targeting grid**: 0.1 intervals (0.1, 0.2, ..., 0.9) forming an 81-point grid

**Note**: Density is evaluated on a coarser 10×10 grid for computational efficiency, while survival functions are evaluated on a finer 200×200 grid.

### 1.3 Performance Metrics
For each method and grid point, we computed:
- **Bias**: Mean fitted density - True density
- **MSE**: Mean squared error
- **Standard deviation**: Variability across simulations
- **Coverage**: Proportion of simulations where the true value falls within the nominal 95% CI (mean ± 1.96 × std)

### 1.4 Region Definitions
- **Overall**: All 100 grid points (10×10)
- **Target Grid**: Grid points within 0.02 tolerance of targeting values (0.1, 0.2, ..., 0.9)
- **Center Region**: Points where T1 < 0.8 OR T2 < 0.8
- **Tail Region**: Points where T1 > 0.8 OR T2 > 0.8

---

## 2. Overall Performance Summary

| Method | Abs Bias | Std Dev | MSE | Coverage |
|--------|----------|---------|-----|----------|
| **Initial_1** | 0.3576 | 0.1248 | 0.2507 | 0.350 |
| **Final_5** | 0.1201 | 0.2070 | 0.0825 | 0.925 |
| **NPMLE_Targeted** | 0.1269 | 0.5353 | 0.4508 | 0.950 |

### Key Observations

1. **Final_5 achieves the best bias-variance tradeoff**:
   - Substantially lower bias than Initial_1 (0.1201 vs 0.3576)
   - Lowest MSE among all methods (0.0825)
   - Excellent coverage (92.5%)

2. **NPMLE_Targeted shows high variance**:
   - Similar bias to Final_5 (0.1269 vs 0.1201)
   - Dramatically higher standard deviation (0.5353 vs 0.2070)
   - This high variance leads to the highest MSE (0.4508) despite good coverage (95.0%)
   - The targeting procedure may be introducing additional variability

3. **Initial_1 severely underestimates the true density**:
   - Highest absolute bias (0.3576)
   - Poor coverage (35.0%), indicating systematic underestimation
   - Not adequate for practical use without further EM iterations

---

## 3. Performance on Target Grid

The target grid consists of 4 points near the targeting values used in the NPMLE procedure.

| Method | Abs Bias | Std Dev | MSE | Coverage |
|--------|----------|---------|-----|----------|
| **Initial_1** | 0.2218 | 0.0817 | 0.0792 | 0.338 |
| **Final_5** | 0.0332 | 0.0803 | 0.0086 | 0.925 |
| **NPMLE_Targeted** | 0.0642 | 0.2209 | 0.0579 | 0.913 |

### Key Observations

1. **Final_5 excels on the target grid**:
   - Extremely low bias (0.0332) and MSE (0.0086)
   - Maintains excellent coverage (92.5%)
   - Demonstrates that the EM procedure effectively targets these regions

2. **NPMLE_Targeted shows improvement over overall performance**:
   - Lower bias (0.0642) compared to overall (0.1269)
   - But still 2× higher variance than Final_5 (0.2209 vs 0.0803)
   - The targeting procedure appears to be working but with added noise

3. **Target grid performance better than overall**:
   - All methods perform better on the target grid than overall
   - This suggests the targeting strategy is effective at these specific points

---

## 4. Regional Performance Analysis

### 4.1 Center Region (T1 < 0.8 OR T2 < 0.8)

| Method | Abs Bias | Std Dev | MSE | Coverage |
|--------|----------|---------|-----|----------|
| **Initial_1** | 0.3521 | 0.1240 | 0.2503 | 0.362 |
| **Final_5** | 0.1213 | 0.2112 | 0.0851 | 0.925 |
| **NPMLE_Targeted** | 0.1309 | 0.5518 | 0.4684 | 0.950 |

### 4.2 Tail Region (T1 > 0.8 OR T2 > 0.8)

| Method | Abs Bias | Std Dev | MSE | Coverage |
|--------|----------|---------|-----|----------|
| **Initial_1** | 0.4247 | 0.1279 | 0.2318 | 0.075 |
| **Final_5** | 0.1283 | 0.1630 | 0.0567 | 0.900 |
| **NPMLE_Targeted** | 0.1215 | 0.4327 | 0.2751 | 0.925 |

### Key Observations

1. **Initial_1 degrades significantly in tail regions**:
   - Bias increases from 0.3521 (center) to 0.4247 (tail)
   - Coverage drops dramatically to 7.5% in tails
   - Clear evidence of systematic underestimation in low-density regions

2. **Final_5 maintains stable performance across regions**:
   - Similar bias in center (0.1213) and tail (0.1283)
   - Lower variance in tails (0.1630) than in center (0.2112)
   - Best MSE in both regions

3. **NPMLE_Targeted shows improved bias in tails**:
   - Slightly lower bias in tails (0.1215) compared to center (0.1309)
   - But variance remains high in both regions
   - Better coverage in tails (92.5%) than center (95.0%)

---

## 5. Comparative Analysis

### 5.1 Bias Patterns

- **Initial_1**: Exhibits consistent **positive bias** across all regions, systematically overestimating density values. This is expected after only one EM iteration, as the algorithm hasn't converged.

- **Final_5**: Shows **substantially reduced bias** compared to Initial_1, with relatively uniform bias patterns across the evaluation grid. The 5 EM iterations provide good convergence.

- **NPMLE_Targeted**: Demonstrates **comparable bias** to Final_5 overall, but with some spatial variation. The targeting procedure appears to slightly increase bias compared to standard EM.

### 5.2 Variance-Coverage Tradeoff

- **Initial_1**: Low variance (0.1248) but **poor coverage** (35.0%), indicating that the low variance is artificial due to systematic bias.

- **Final_5**: Moderate variance (0.2070) with **excellent coverage** (92.5%), achieving the optimal balance for reliable inference.

- **NPMLE_Targeted**: Very high variance (0.5353) with **excellent coverage** (95.0%). The high variance suggests that the targeting procedure introduces substantial sampling variability, possibly due to the additional estimation steps involved.

### 5.3 MSE Decomposition

The MSE can be decomposed as MSE = Bias² + Variance. Examining this tradeoff:

- **Initial_1**: MSE dominated by bias² (0.1279 out of 0.2507)
- **Final_5**: Better bias-variance balance (bias² ≈ 0.0144, variance ≈ 0.0428)
- **NPMLE_Targeted**: MSE dominated by variance (variance ≈ 0.2866 out of 0.4508)

This analysis clearly shows that **Final_5 achieves the best overall tradeoff**.

---

## 6. Recommendations

### 6.1 Method Selection

1. **For general density estimation**: Use **Final_5**
   - Best MSE performance
   - Excellent coverage
   - Stable across all regions
   - Lower computational overhead than NPMLE_Targeted

2. **For targeted inference**: Consider **Final_5** over NPMLE_Targeted
   - Final_5 actually outperforms NPMLE_Targeted on the target grid
   - Much lower variance leads to more precise estimates
   - The additional complexity of NPMLE_Targeted doesn't appear justified by the results

3. **Avoid Initial_1 for inference**:
   - Severe bias and poor coverage make it unsuitable
   - Use only as an initialization for further EM iterations

### 6.2 Further Investigation

1. **Investigate NPMLE_Targeted high variance**:
   - Examine whether the targeting procedure is unstable
   - Consider variance reduction techniques (e.g., cross-fitting, sample splitting)
   - Test with different targeting grids or strategies

2. **Optimize EM iterations**:
   - Current results suggest 5 iterations is sufficient
   - Could investigate convergence criteria to adaptively choose iterations

3. **Extend to larger sample sizes**:
   - Evaluate whether NPMLE_Targeted variance decreases at faster rate
   - Assess whether targeting benefits emerge with larger n

---

## 7. Conclusions

This simulation study provides strong evidence that:

1. **Final_5 (EM with 5 iterations) is the recommended method** for bivariate density estimation with censored data, achieving the best balance of bias, variance, and MSE.

2. **NPMLE_Targeted introduces substantial additional variance** without compensating gains in bias reduction, leading to higher MSE despite good coverage properties.

3. **Regional performance is relatively consistent** across center and tail regions for Final_5 and NPMLE_Targeted, while Initial_1 shows clear degradation in tails.

4. **The targeting strategy itself appears effective** (as evidenced by improved target grid performance), but the NPMLE_Targeted implementation may need refinement to reduce variance.

Future work should focus on understanding and potentially reducing the variance of NPMLE_Targeted while maintaining its theoretical advantages.

---

## Appendix: Visualizations

The accompanying Jupyter notebook ([censored_NPMLE_EM_Density_Summary.ipynb](censored_NPMLE_EM_Density_Summary.ipynb)) contains detailed visualizations including:

1. **Density heatmaps**: Side-by-side comparison of true density vs. fitted densities for each method
2. **Bias heatmaps**: Spatial patterns of bias across the evaluation grid
3. **Performance tables**: Detailed breakdowns by region and method

All heatmaps use consistent color scales to facilitate direct comparison between methods.

---

**Report Generated**: 2025-12-07
**Data Source**: FineGridEval_40_censored_density_results_TargetingGRID0.1_nSamples500.pkl
**Analysis Code**: censored_NPMLE_EM_Density_Summary.ipynb
