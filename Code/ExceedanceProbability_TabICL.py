import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tabicl import TabICLRegressor  # <--- Changed import to TabICL

base_dataset = "SpurDike"
model = "TabICL"
param_name = "VVc" #yl|VVc|Fd50
name_map = {
    "dsl": "$d_s / l$",
    "yl": "$y / l$",
    "VVc": "$V / V_c$",
    "ld50": "$l / d_{50}$",
    "Fd50": "$Fd_{50}$",
}

print(f"model: {model}, param: {param_name}")


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

# ---- Define critical threshold ----
d_critical = np.percentile(y, 90)
print(f"Critical threshold d_critical = {d_critical}")

# ---- Initialize and fit TabICL ----
reg = TabICLRegressor()
reg.fit(X, y)

# Predict full distribution using TabICL
preds = reg.predict(X, output_type="quantiles")

# ---- Extract quantiles and levels for TabICL ----
q_vals = np.array(preds)           

# Robust check: If the first dimension matches the number of rows in X,
# then TabICL returned it as (n_samples, n_quantiles). Transpose it!
if q_vals.shape[0] == X.shape[0]:
    q_vals = q_vals.T  # Now safely guaranteed to be (n_quantiles, n_samples)

n_quantiles, n_samples = q_vals.shape

# Define quantile levels matching TabICL's exact dynamic resolution
q_levels = np.linspace(1 / (n_quantiles + 1), n_quantiles / (n_quantiles + 1), n_quantiles)

# ---- Compute exceedance probability ----
P_exceed = np.zeros(n_samples)

for i in range(n_samples):
    qs = q_vals[:, i]

    # Ensure monotonicity (important for interpolation stability)
    qs_sorted_idx = np.argsort(qs)
    qs_sorted = qs[qs_sorted_idx]
    ql_sorted = q_levels[qs_sorted_idx]

    # Interpolate CDF: P(ds <= d_critical)
    prob_leq = np.interp(
        d_critical,
        qs_sorted,
        ql_sorted,
        left=0.0,
        right=1.0
    )

    # Convert to exceedance probability
    P_exceed[i] = 1 - prob_leq

# ---- Summary statistics ----
print(f"Critical threshold d_critical = {d_critical}")
print(f"Mean exceedance probability: {P_exceed.mean():.3f}")
print(f"Max exceedance probability: {P_exceed.max():.3f}")
print(f"Min exceedance probability: {P_exceed.min():.3f}")


# ===============================
# Plot: Exceedance Probability vs param
# ===============================

param = X[param_name].values if hasattr(X, "columns") else X[:, 0]

plt.figure(figsize=(4, 4), dpi=300)
plt.scatter(param, P_exceed, c="purple", alpha=0.7)

# Risk thresholds
plt.axhline(0.1, linestyle="--", linewidth=1, label="10% risk")
plt.axhline(0.5, linestyle="--", linewidth=1, label="50% risk")

# ---------------------------------------------------------
# Add vertical dashed lines based on param_name (in log10 scale)
# ---------------------------------------------------------
if param_name == "VVc":
    plt.axvline(x=1, color="red", linestyle="--", linewidth=1.5, alpha=0.8, label="$V/V_c ≈ 1$")
elif param_name == "Fd50":
    plt.axvline(x=2.7, color="red", linestyle="--", linewidth=1.5, alpha=0.8, label="$Fd_{50} ≈ 2.7$")

plt.xlabel(f"{name_map[param_name]}")
plt.ylabel("$P(d_s / l > d_{critical})$")
plt.title(f"Exceedance Probability vs. {name_map[param_name]}", fontsize=10)

plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

# Save and show
plt.savefig(f"../Figures/Exceedance_Probability_vs_{param_name}_{base_dataset}_{model}.png", dpi=300, bbox_inches="tight")
plt.show()