base_dataset = "SpurDike"

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tabicl import TabICLRegressor  # <--- Changed import to TabICL

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

# ---- Initialize and fit TabICL ----
reg = TabICLRegressor()
reg.fit(X, y)

# Predict full distribution using TabICL
preds = reg.predict(X, output_type="quantiles")

# ---- Extract quantiles and handle shape dynamically ----
quantiles = np.array(preds)           

# If the first dimension matches dataset length, transpose to shape (n_quantiles, n_samples)
if quantiles.shape[0] == X.shape[0]:
    quantiles = quantiles.T  

n_quantiles, n_samples = quantiles.shape
print("Quantiles shape:", quantiles.shape)

# ---- Map dynamic index locations for Q10, Q50, and Q90 ----
# This scales cleanly regardless of the exact dense grid size returned by TabICL
idx_10 = int(np.round(0.10 * (n_quantiles - 1)))
idx_50 = int(np.round(0.50 * (n_quantiles - 1)))
idx_90 = int(np.round(0.90 * (n_quantiles - 1)))

q10 = quantiles[idx_10]   # Exact 10th percentile array across all samples
q50 = quantiles[idx_50]   # Exact 50th percentile (Median) array
q90 = quantiles[idx_90]   # Exact 90th percentile array

# ===============================
# Probabilistic metrics
# ===============================

# Convert to numpy arrays (safety)
y_true = np.array(y)
lower = np.array(q10)
upper = np.array(q90)

# PICP: proportion of observations inside interval
inside_interval = (y_true >= lower) & (y_true <= upper)
PICP = np.mean(inside_interval)

# MPIW: average width of interval
MPIW = np.mean(upper - lower)

print(f"PICP (90% interval): {PICP:.3f}")
print(f"MPIW: {MPIW:.3f}")

MPIW_norm = MPIW / (y_true.max() - y_true.min())
print(f"Normalized MPIW: {MPIW_norm:.3f}")


######## Probabilistic Prediction Plot
plt.figure(figsize=(4, 4), dpi=300)

# Scatter: median prediction
plt.scatter(y, q50, c="tab:blue", alpha=0.7, label="Median prediction")

# Prediction intervals
plt.vlines(
    y,
    q10,
    q90,
    color="gray",
    alpha=0.4,
    linewidth=1.5,
    label="90% prediction interval"
)

# 1:1 reference line
lims = [min(y.min(), q10.min()), max(y.max(), q90.max())]
plt.plot(lims, lims, "k--", linewidth=1)

plt.xlabel("Observed scour depth $d_s / l$")
plt.ylabel("Predicted scour depth $\\hat{d}_s / l$")
plt.title(f"Probabilistic Prediction", fontsize=10)
plt.legend()

plt.text(
    0.05, 0.95,
    f"PICP = {PICP:.2f}\nMPIW = {MPIW:.2f}",
    transform=plt.gca().transAxes,
    fontsize=9,
    verticalalignment='top',
    bbox=dict(boxstyle="round", facecolor="white", alpha=0.7)
)

plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"../Figures/Uncertainty_Probabilistic_Prediction_{base_dataset}_TabICL.png", dpi=300, bbox_inches="tight")
plt.show()


########### Uncertainty vs Flow intensity
# Compute uncertainty width
uncertainty_width = q90 - q10

param_name = "VVc"  # "yl" or "Fd50" | "VVc" --- IGNORE ---
name_map = {
    "dsl": "$d_s / l$",
    "yl": "$y / l$",
    "VVc": "$V / V_c$",
    "ld50": "$l / d_{50}$",
    "Fd50": "$Fd_{50}$",
}

param = X[param_name].values if hasattr(X, "columns") else X[:, 0]

plt.figure(figsize=(4, 4), dpi=300)
plt.scatter(param, uncertainty_width, c="tab:red", alpha=0.7)

# ---------------------------------------------------------
# Add vertical dashed line based on param_name
# ---------------------------------------------------------
if param_name == "VVc":
    plt.axvline(
        x=1,
        color="red",
        linestyle="--",
        linewidth=1.5,
        alpha=0.8,
        label="$V/V_c ≈ 1$",
    )
elif param_name == "Fd50":
    plt.axvline(
        x=2.7,
        color="red",
        linestyle="--",
        linewidth=1.5,
        alpha=0.8,
        label="$Fd_{50} ≈ 2.7$",
    )

plt.xlabel(f"{name_map[param_name]}")
plt.ylabel("Prediction interval width $(Q_{0.9} - Q_{0.1})$")
plt.title(f"Uncertainty of Scour Prediction vs. {name_map[param_name]}", fontsize=10)
plt.grid(alpha=0.3)

# Show legend if a threshold line was added
if param_name in ["VVc", "Fd50"]:
    plt.legend()

plt.tight_layout()
plt.savefig(f"../Figures/Uncertainty_Prediction_vs_{param_name}_{base_dataset}_TabICL.png", dpi=300, bbox_inches="tight")
plt.show()