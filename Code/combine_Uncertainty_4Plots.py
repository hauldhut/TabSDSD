from PIL import Image
import matplotlib.pyplot as plt

base_dataset = "SpurDike"
model = "TabICL" #TabPFN|TabICL
print(f"model: {model}, param: yl, Fd50, VVc")

# Load individual figures
img1 = Image.open(f"../Figures/Uncertainty_Probabilistic_Prediction_{base_dataset}_{model}.png")
img2 = Image.open(f"../Figures/Uncertainty_Prediction_vs_yl_{base_dataset}_{model}.png")
img3 = Image.open(f"../Figures/Uncertainty_Prediction_vs_Fd50_{base_dataset}_{model}.png")
img4 = Image.open(f"../Figures/Uncertainty_Prediction_vs_VVc_{base_dataset}_{model}.png")

gap = 50

# Row 1 dimensions: img1 (A) and img2 (B)
row1_width = img1.width + img2.width + gap
row1_height = max(img1.height, img2.height)

# Row 2 dimensions: img3 (C) and img4 (D)
row2_width = img3.width + img4.width + gap
row2_height = max(img3.height, img4.height)

# Total canvas dimensions for 2 rows x 2 columns
width = max(row1_width, row2_width)
height = row1_height + gap + row2_height

# Create new combined blank image
combined = Image.new("RGB", (width, height), "white")

# Paste Row 1
combined.paste(img1, (0, 0))
combined.paste(img2, (img1.width + gap, 0))

# Paste Row 2
combined.paste(img3, (0, row1_height + gap))
combined.paste(img4, (img3.width + gap, row1_height + gap))

# Plot combined image using Matplotlib
plt.figure(figsize=(8, 8), dpi=300)
plt.imshow(combined)
plt.axis("off")

# Add panel labels (A, B, C, D)
# Row 1 labels
plt.text(10, 25, "A", fontsize=11, fontweight="bold")
plt.text(img1.width + gap + 10, 25, "B", fontsize=11, fontweight="bold")

# Row 2 labels
y_offset_row2 = row1_height + gap
plt.text(10, y_offset_row2 + 25, "C", fontsize=11, fontweight="bold")
plt.text(img3.width + gap + 10, y_offset_row2 + 25, "D", fontsize=11, fontweight="bold")

plt.tight_layout()

# Save the multi-panel figure
plt.savefig(
    f"../Figures/Uncertainty_Prediction_vs_yl_Fd50_VVc_{base_dataset}_{model}_Final.png",
    dpi=600,
    bbox_inches="tight"
)
plt.show()