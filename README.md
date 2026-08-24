# TabSDSD: Probabilistic Spur-Dike Scour Prediction Using an In-Context Learning-Based Tabular Foundation Model


- TabICL accurately predicts normalized equilibrium scour around spur dikes.
- TabICL outperforms empirical equations and advanced machine-learning models.
- SHAP reveals dominant predictors and nonlinear scour relationships.
- Probabilistic predictions quantify condition-dependent scour uncertainty.
- Exceedance probabilities enable uncertainty-aware, risk-informed scour assessment.

![TabICL for spur-dike scour depth prediction](https://github.com/hauldhut/TabSDSD/blob/main/TabSDSD.png)

### Performance comparison between TabICLs and advanced machine learning-based models


| Model                                   |             CC |             R² |           RMSE |            MAE |           MAPE |             MBE |
| --------------------------------------- | -------------: | -------------: | -------------: | -------------: | -------------: | --------------: |
| **TabICL**                              | 0.981 (±0.014) | 0.957 (±0.030) | 0.159 (±0.066) | 0.097 (±0.032) | 0.187 (±0.106) | −0.006 (±0.033) |
| *Probabilistic models*                  |                |                |                |                |                |                 |
| TabPFN                                  | 0.979 (±0.015) | 0.954 (±0.032) | 0.165 (±0.073) | 0.101 (±0.037) | 0.194 (±0.111) | −0.009 (±0.036) |
| NGBoost                                 | 0.937 (±0.039) | 0.868 (±0.081) | 0.288 (±0.129) | 0.192 (±0.053) | 0.288 (±0.113) | −0.018 (±0.078) |
| *Deterministic boosting-based models*   |                |                |                |                |                |                 |
| AdaBoost                                | 0.910 (±0.037) | 0.800 (±0.087) | 0.353 (±0.072) | 0.270 (±0.039) | 0.432 (±0.089) |  0.040 (±0.090) |
| XGBoost                                 | 0.939 (±0.044) | 0.874 (±0.088) | 0.279 (±0.137) | 0.172 (±0.058) | 0.258 (±0.119) | −0.027 (±0.060) |
| CatBoost                                | 0.952 (±0.033) | 0.896 (±0.070) | 0.254 (±0.120) | 0.165 (±0.054) | 0.255 (±0.122) | −0.014 (±0.064) |

## Repo structure
- **Data**: Contains all datasets 
- **Code**: Contains all source code to reproduce all the results

## How to run
- Follow instructions in the folders **Data** and **Code** to run
