base_dataset = "SpurDike"
model = "TabICL" #TabICL|TabPFN

#Load shap_values
import pickle
import numpy as np

shapfile = f"../Results/shap_values_SpurDike_all_{model}.pkl"

with open(shapfile, "rb") as f:
    shap_values = pickle.load(f)

print(shap_values.feature_names)

name_map = {
    "dsl": "$d_s / l$",
    "yl": "$y / l$",
    "VVc": "$V / V_c$",
    "ld50": "$l / d_{50}$",
    "Fd50": "$Fd_{50}$",
}
shap_values.feature_names = [
    name_map.get(name, name) for name in shap_values.feature_names
]

import shap
import matplotlib.pyplot as plt

# shap_values.values: (n_samples, n_features)
# shap_values.data:   feature matrix
# shap_values.feature_names

base_fontsize = 14

# 1️⃣ Global importance
plt.figure(figsize=(8, 4), dpi=300)
shap.plots.bar(shap_values, show=False)
plt.title("Global SHAP Feature Importance", fontsize=base_fontsize+2)
plt.xlabel("mean(|SHAP value|)", fontsize=base_fontsize)
# plt.ylabel("Predicted scour depth $\\hat{d}_s$")

# Tick label font size
plt.xticks(fontsize=base_fontsize)
plt.yticks(fontsize=base_fontsize)

plt.tight_layout()
plt.savefig(f"../Figures/SHAP_global_{base_dataset}_{model}.png", dpi=300)
plt.show()


###############################
import matplotlib.pyplot as plt

base_fontsize = 14

plt.figure(figsize=(4, 8), dpi=300)
# shap.plots.beeswarm(shap_values, show=False)

shap.plots.beeswarm(
    shap_values,
    show=False,
    max_display=10,
    color_bar=True
)

# Get current figure and axes
fig = plt.gcf()
axes = fig.axes

# Axis 0: main beeswarm plot
axes[0].set_title("SHAP Summary Plot", fontsize=base_fontsize + 2)
axes[0].set_xlabel(
    "SHAP value (impact on model output)",
    fontsize=base_fontsize
)
axes[0].tick_params(axis="both", labelsize=base_fontsize)

# Make points larger and slightly transparent
for coll in axes[0].collections:
    coll.set_alpha(0.7)
    coll.set_sizes([30])   # increase size (default is small)

# Axis 1: colorbar (feature value)
axes[1].set_ylabel(
    "Feature value",
    fontsize=base_fontsize
)
axes[1].tick_params(labelsize=base_fontsize)

plt.tight_layout()
plt.savefig(f"../Figures/SHAP_summary_{base_dataset}_{model}.png", dpi=300)
plt.show()

# -------------------------------------------------------------
# 3. Create Scatter Plots in One Row (Sorted by Importance)
# -------------------------------------------------------------
base_fontsize = 14
num_features = len(shap_values.feature_names)

# Calculate global feature importance: mean(|SHAP values|) across all samples
mean_abs_shap = np.abs(shap_values.values).mean(axis=0)

# Get feature indices sorted by importance (descending)
sorted_indices = np.argsort(mean_abs_shap)[::-1]

# Dynamically set figure width based on the number of features
fig, axes = plt.subplots(
    nrows=1, ncols=num_features, figsize=(4.5 * num_features, 4.5), dpi=300
)

# Handle single feature case gracefully
if num_features == 1:
    axes = [axes]

# Iterate through the sorted indices to draw plots in descending importance
for i, feature_idx in enumerate(sorted_indices):
    feature_name = shap_values.feature_names[feature_idx]
    ax = axes[i]

    # Draw scatter plot on the given axis
    shap.plots.scatter(
        shap_values[:, feature_name],
        color=shap_values[:, feature_name],  # Colors points by feature value
        ax=ax,
        show=False,
    )

    # ---------------------------------------------------------
    # Add vertical dashed lines & explicit x-axis tick marks
    # ---------------------------------------------------------
    if feature_name in ["$V / V_c$", "VVc"]:
        ax.axvline(x=1, color="red", linestyle="--", linewidth=1.5, alpha=0.8)
        # ax.set_xticks([1, 2, 3, 4, 5])
        
    elif feature_name in ["$Fd_{50}$", "Fd50"]:
        ax.axvline(x=2.7, color="red", linestyle="--", linewidth=1.5, alpha=0.8)
        # ax.set_xticks([5, 10, 15, 20])

    # Styling for individual subplot
    ax.tick_params(axis="both", labelsize=base_fontsize)
    ax.set_xlabel(feature_name, fontsize=base_fontsize + 1)

    # Keep y-label on the first subplot only to save space
    if i == 0:
        ax.set_ylabel("SHAP value", fontsize=base_fontsize + 1)
    else:
        ax.set_ylabel("")

plt.tight_layout()
plt.savefig(
    f"../Figures/SHAP_scatter_row_{base_dataset}_{model}.png",
    dpi=300,
    bbox_inches="tight",
)
plt.show()