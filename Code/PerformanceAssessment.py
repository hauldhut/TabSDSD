# Data Science & Visualization
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

# Other ML Models
from catboost import CatBoostClassifier, CatBoostRegressor

# Notebook UI/Display
from IPython.display import Markdown, display
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.rule import Rule
from sklearn.compose import make_column_selector, make_column_transformer

# Scikit-Learn: Data & Preprocessing
from sklearn.datasets import fetch_openml, load_breast_cancer

# Scikit-Learn: Models
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, AdaBoostRegressor
from sklearn.metrics import mean_squared_error, roc_auc_score
from sklearn.model_selection import (
    KFold,
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from xgboost import XGBClassifier, XGBRegressor
from ngboost import NGBRegressor
from ngboost.distns import Normal

# This transformer will be used to handle categorical features for the baseline models
column_transformer = make_column_transformer(
    (
        OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
        make_column_selector(dtype_include=["object", "category"]),
    ),
    remainder="passthrough",
)

base_dataset = "SpurDike"

import pandas as pd

# Read the tab-separated file
dfSD = pd.read_csv("../Data/SpurDike.tsv", sep="\t")
print(dfSD.shape)
# Show first rows
print(dfSD.head())
dfSD["Source"].value_counts()

selected_Source = ["Dey&Barbhuiya2005","Nasrollahi2008","Coleman2003","Pandey2016","Lim1997"]

dfSD_filtered = dfSD[dfSD["Source"].isin(selected_Source)]
print(dfSD_filtered.shape)

X, y = dfSD_filtered.drop(columns=["dsl", "Source"]), dfSD_filtered["dsl"]

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, cross_validate
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    mean_absolute_percentage_error,
    make_scorer,
)

# RMSE (negative because sklearn assumes higher = better)
rmse_scorer = make_scorer(
    lambda y_true, y_pred: np.sqrt(mean_squared_error(y_true, y_pred)),
    greater_is_better=False,
)

# Pearson correlation coefficient
corr_scorer = make_scorer(
    lambda y_true, y_pred: np.corrcoef(y_true, y_pred)[0, 1]
)
# MBE (Mean Bias Error)
mbe_scorer = make_scorer(
    lambda y_true, y_pred: np.mean(y_pred - y_true),
    greater_is_better=False   # closer to 0 is better
)

from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.linear_model import LinearRegression

from tabpfn import TabPFNClassifier, TabPFNRegressor
from tabicl import TabICLRegressor

# ─── FIX FOR NEwER SCIKIT-LEARN VERSIONS ──────────────────────────────────
# Tells the Pipeline that NGBoost is successfully fitted if it has base_models
NGBRegressor.__sklearn_is_fitted__ = lambda self: hasattr(self, "base_models") and len(self.base_models) > 0

models = [
    ("TabPFN", TabPFNRegressor(random_state=42)),
    ("TabICL", TabICLRegressor(random_state=42)),
    (
        "NGBoost",
        make_pipeline(
            column_transformer,
            NGBRegressor(random_state=42, verbose=False),  # added NGBoost here
        ),
    ),
    (
        "AdaBoost",
        make_pipeline(
            column_transformer,
            AdaBoostRegressor(random_state=42),
        ),
    ),
    (
        "XGBoost",
        make_pipeline(
            column_transformer,
            XGBRegressor(random_state=42),
        ),
    ),
    (
        "CatBoost",
        make_pipeline(
            column_transformer,
            CatBoostRegressor(random_state=42, verbose=0),
        ),
    ),

]


n_splits = 10
cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)

# @title
import numpy as np
import pandas as pd

scoring = {
    "CC": corr_scorer,
    "R2": "r2",
    "RMSE": "neg_root_mean_squared_error",
    "MAE": "neg_mean_absolute_error",
    "MAPE": "neg_mean_absolute_percentage_error",
    "MBE": mbe_scorer,
}

results = []

def mean_std_str(values, negate=False):
    """Return formatted string: mean (± std), rounded to 3 decimals"""
    if negate:
        values = -values
    mean = np.mean(values)
    std = np.std(values, ddof=1)  # sample std
    return f"{mean:.3f} (±{std:.3f})"

for name, model in models:
    print(f"Evaluating {name}...")
    cv_results = cross_validate(
        model,
        X,
        y,
        cv=cv,
        scoring=scoring,
        n_jobs=1,
        verbose=1,
    )

    results.append({
        "Model": name,
        "CC": mean_std_str(cv_results["test_CC"]),
        "R2": mean_std_str(cv_results["test_R2"]),
        "RMSE": mean_std_str(cv_results["test_RMSE"], negate=True),
        "MAE": mean_std_str(cv_results["test_MAE"], negate=True),
        "MAPE": mean_std_str(cv_results["test_MAPE"], negate=True),
        "MBE": mean_std_str(cv_results["test_MBE"], negate=True),
    })

df_results = pd.DataFrame(results)
df_results.to_csv("../Results/PerformanceComparison.tsv", sep="\t", index=False)

df_results.sort_values("RMSE")
print(df_results)