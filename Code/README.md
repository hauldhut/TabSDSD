<img width="468" height="28" alt="image" src="https://github.com/user-attachments/assets/a195b6fa-0158-4c51-a154-8de2723d5d4d" />
# This folder contains all source code used in the manuscript

## Environment Setup
- **Create conda environment**
  - conda create -n TabSDSD python=3.12
  - conda activate TabSDSD
  - pip install -r requirements.txt

## Experiments
The analysis follows this order: Performance assessment → Physical interpretability → Uncertainty quantification → Exceedance probability → Figure Creation

- **1. Performance assessment**
  - *EmpiricalEquations.py*: For assessing the prediction performance of empirical equations: Dey2005 (Dey and Barbhuiya, 2005), Pandey2016 (Pandey, et al., 2016), and Singh2024 (Singh and Minocha, 2024)
  - *PerformanceAssessment.py*: For performance assessment and comparison between TabICL and two probabilistic models (TabPFN, NGBoost) and three deterministic boosting-based models (CatBoost, XGBoost, and AdaBoost)

- **2. Physical interpretability**
  - *Interpretability.py*: For generating SHAP values
  - *Interpretability_drawFigs.py*: For draw SHAP plots from SHAP values

- **3. Uncertainty quantification**
  - *UncertaintyQuantification.py*: For uncertainty quantification. Change param_name = "VVc" or "yl" or "Fd50"

- **4. Exceedance probability**
  - *ExceedanceProbability.py*: For exceedance probability analysis. Change param_name = "VVc" or "yl" or "Fd50" 

## Figure Creation
  - *combine_Interpretability_3Plots.py*: To create Figure 3
  - *combine_Uncertainty_4Plots.py*: To create Figure 4
  - *combine_Exceedance_3Plots.py*: To create Figure 5
