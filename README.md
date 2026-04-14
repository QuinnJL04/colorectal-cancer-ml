# Colorectal cancer ML

## Problem Statement
This project investigates colon cancer prediction using three complementary machine learning approaches learned in the course. First, a convolutional neural network will be used to classify colorectal histopathology images as cancerous or non-cancerous. Second, a Bayesian regression model will be applied to evaluate the relationship between socioeconomic and environmental factors, such as income level, food access, and healthcare availability, and colorectal cancer incidence rates at the population level. Third, a time series model will be implemented to analyze and forecast trends in colorectal cancer incidence over time. By integrating image-based classification, probabilistic socioeconomic modeling, and temporal forecasting, the project aims to provide both individual-level diagnostic insight and broader public health risk analysis.

## Proposed Project and Approaches
This project integrates three complementary modeling approaches to examine colon cancer prediction from multiple perspectives. The CNN-based model addresses individual-level image classification, building on established research in medical computer vision. The Bayesian regression model provides a probabilistic framework for analyzing medical background disparities in cancer incidence, offering interpretable credible intervals for risk factors. The time series model contributes a forecasting dimension, allowing for analysis of incidence trends over time.
By combining these approaches, the project aims to provide a comprehensive view of colon cancer prediction that integrates clinical imaging, socioeconomic risk factors, and temporal dynamics. This multi-method framework aligns with prior literature while extending it by connecting diagnostic modeling with broader population-level risk analysis.

## Repository layout

| Path | Contents |
|------|----------|
| `time-series/ARIMA.ipynb` | Univariate ARIMA forecast of US incidence (SEER-based series in `data/`) |
| `bayesian/` | Bayesian analysis notebooks and sample data |
| `data/` | CSV datasets used by the notebooks |

## Methods (summary)

- **Time series:** ARIMA on annual, age-adjusted colorectal cancer **incidence**.
- **Bayesian:** Population-level modeling with uncertainty.
- **CNN:** Identifying colon cancer based on colon cancer image data. 

## Data sources

Links are starting points; cite the exact extract and years you used in your report.

- **SEER / incidence context:** [NCI SEER Stat Facts — colorectal cancer](https://seer.cancer.gov/statfacts/html/colorect.html)  
- **Zenodo (histopathology images):** [Zenodo record 1214456](https://zenodo.org/records/1214456)  
- **OpenDataBay dataset:** [Premium dataset listing](https://www.opendatabay.com/data/premium/ae2aba99-491d-45a1-a99e-7be14927f4af)  
- **Kaggle — global colorectal dataset:** [Kaggle dataset](https://www.kaggle.com/datasets/ankushpanday2/colorectal-cancer-global-dataset-and-predictions)  

## Requirements

Use a Python environment with at least: `pandas`, `numpy`, `matplotlib`, `statsmodels`, `scikit-learn`.

## Authors

Quinn Louie, Vijay Chinnam, Jake Wu-Chen