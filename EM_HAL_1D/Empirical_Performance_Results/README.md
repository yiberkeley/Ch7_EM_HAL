# Empirical Performance Results

This folder contains empirical performance analysis results for both Interval Censored (IC) and Right Censored (RC) cases.

## Analysis Overview

The analysis evaluates the empirical performance of density and survival estimates using:

- **Oracle Coverage**: Coverage computed using the sample standard deviation across simulations as the oracle standard error (NOT delta method CI coverage)
- **Bias**: Mean estimate - true value
- **Variance**: Empirical variance across simulations
- **MSE**: Mean squared error

**Important**: All metrics are computed only for evaluation points within the **inner 99% range** of the true distribution **[0.242, 0.758]** to avoid tail regions where sparse observations can cause estimation issues.

## Folder Structure

```
Empirical_Performance_Results/
├── IC/                                           # Interval Censored results
│   ├── IC_Combined_Empirical_Performance_Summary.csv
│   ├── IC_Density_Empirical_Performance_Summary.csv
│   ├── IC_Density_Performance_Smooth0.png
│   ├── IC_Density_Performance_Smooth1.png
│   ├── IC_Survival_Empirical_Performance_Summary.csv
│   ├── IC_Survival_Performance_Smooth0.png
│   └── IC_Survival_Performance_Smooth1.png
│
└── RC/                                           # Right Censored results
    ├── RC_Combined_Empirical_Performance_Summary.csv
    ├── RC_Density_Empirical_Performance_Summary.csv
    ├── RC_Density_Performance_Smooth0.png
    ├── RC_Density_Performance_Smooth1.png
    ├── RC_Survival_Empirical_Performance_Summary.csv
    ├── RC_Survival_Performance_Smooth0.png
    └── RC_Survival_Performance_Smooth1.png
```

## File Descriptions

### CSV Files

Each folder contains three summary tables:

1. **`*_Density_Empirical_Performance_Summary.csv`**: Performance metrics for density estimates
2. **`*_Survival_Empirical_Performance_Summary.csv`**: Performance metrics for survival estimates
3. **`*_Combined_Empirical_Performance_Summary.csv`**: Combined table with both density and survival results

**Columns in summary tables:**
- `Estimate Type`: Density or Survival
- `Sample Size`: Sample size (n)
- `Smooth Order`: Smoothness constraint (0 or 1)
- `N Eval Points`: Number of evaluation points within inner 99% range
- `Mean Abs Bias`: Average absolute bias
- `Mean Variance`: Average variance across evaluation points
- `Mean MSE`: Average mean squared error
- `Median Oracle Coverage (%)`: Median oracle coverage percentage
- `Mean Oracle Coverage (%)`: Mean oracle coverage percentage

### PNG Files

Each folder contains visualization plots for each smooth order:

- **`*_Density_Performance_Smooth*.png`**: Three-panel plots showing bias, MSE, and oracle coverage for density estimates across sample sizes
- **`*_Survival_Performance_Smooth*.png`**: Three-panel plots showing bias, MSE, and oracle coverage for survival estimates across sample sizes

## Key Differences: IC vs RC

### Evaluation Grid
- **IC (Interval Censored)**:
  - Original grid: 100 evaluation points over [0.0, 1.0]
  - Filtered to inner 99%: ~52 evaluation points in [0.242, 0.758]

- **RC (Right Censored)**:
  - Original grid: 20 evaluation points over [0.02, 0.98]
  - Filtered to inner 99%: ~10 evaluation points in [0.242, 0.758]

### Sample Sizes
Both IC and RC cases evaluate sample sizes: 200, 400, 800, 1600, 3200

### Smooth Orders
Both cases evaluate smooth orders: 0 (piecewise constant) and 1 (piecewise linear)

## How to Regenerate Results

Run the corresponding Jupyter notebooks in the parent directory:

- **IC Case**: `Empirical_Performance_IC.ipynb`
- **RC Case**: `Empirical_Performance_RC.ipynb`

Both notebooks will automatically:
1. Filter evaluation points to inner 99% range
2. Compute empirical performance metrics
3. Generate summary tables
4. Create visualization plots
5. Save all outputs to this folder

## Notes

- The inner 99% range [0.242, 0.758] is computed from the true truncated normal distribution with mean=0.5, std=0.1, truncated to [0, 1]
- Oracle coverage is computed for each evaluation point and each simulation by constructing confidence intervals using the sample standard deviation as the oracle standard error
- The 95% target coverage is shown as a red dashed line in coverage plots
