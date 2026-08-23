base_dataset = "SpurDike"

model = "TabICL" #TabPFN|TabICL

from PIL import Image
import matplotlib.pyplot as plt

# Load images
img1 = Image.open(f"../Figures/SHAP_global_{base_dataset}_{model}.png")       # A
img2 = Image.open(f"../Figures/SHAP_summary_{base_dataset}_{model}.png")      # B
img3 = Image.open(f"../Figures/SHAP_scatter_row_{base_dataset}_{model}.png")  # C

gap = 40  # space between panels

# --- Resize images to have same width per column ---
# Make top row images same height
top_height = max(img1.height, img2.height)

# Resize proportionally
def resize_to_height(img, target_h):
    w = int(img.width * target_h / img.height)
    return img.resize((w, target_h))

img1 = resize_to_height(img1, top_height)
img2 = resize_to_height(img2, top_height)

# Bottom image spans full width
total_width = img1.width + img2.width + gap
img3 = img3.resize((total_width, int(img3.height * total_width / img3.width)))

# --- Create canvas ---
total_height = top_height + img3.height + gap

combined = Image.new("RGB", (total_width, total_height), "white")

# Paste top row
combined.paste(img1, (0, 0))
combined.paste(img2, (img1.width + gap, 0))

# Paste bottom row (C)
combined.paste(img3, (0, top_height + gap))

# --- Plot with labels ---
plt.figure(figsize=(8, 10), dpi=300)
plt.imshow(combined)
plt.axis("off")

# Labels
plt.text(10, 25, "A", fontsize=14, fontweight="bold")
plt.text(img1.width + gap + 10, 25, "B", fontsize=14, fontweight="bold")
plt.text(10, top_height + gap + 25, "C", fontsize=14, fontweight="bold")

plt.tight_layout()

# Save final figure
plt.savefig(f"../Figures/SHAP_combined3_{base_dataset}_{model}.png", dpi=300, bbox_inches="tight")
plt.show()