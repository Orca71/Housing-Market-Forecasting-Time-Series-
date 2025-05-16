# Housing Price and Rent Forecasting in Colorado

This project analyzes and predicts **monthly mortgage and rent prices** for single-family homes across three Colorado cities — Denver, Boulder, and Fort Collins. It combines economic, housing, and market indicators with **time series regression**, **Elastic Net**, and **XGBoost models**, and evaluates model assumptions in detail.

> ⚠️ **Note:** This project is still in progress.  
> The code in this repository is being refactored and **does not yet fully reflect the final results and methods** shown in the [Report.pdf](Report.pdf).

---

## Project Summary

- **Forecasted** both mortgage payments and rent using time-series multi-linear regression
- **Addressed all 4 key linear regression assumptions** using log transformation, lag features, and diagnostics
- Used **Elastic Net** to combat multicollinearity and overfitting in complex economic data
- Applied **XGBoost** for robust mortgage prediction with the highest performance
- Compared model performance using metrics like R², MAPE, MAE, and residual analysis

---

## Project Structure

```bash
├── code/                         # In-progress Python notebooks
├── Report.pdf                    # Final report with full methodology and results
└── README.md
