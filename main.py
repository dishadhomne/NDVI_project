import numpy as np
import matplotlib.pyplot as plt

# Load image
img = plt.imread("image.jpg").astype(float)

# Split RGB channels
red = img[:, :, 0]
green = img[:, :, 1]

# NDVI-like formula (simplified)
ndvi = (green - red) / (green + red + 1e-5)

# Vegetation detection
veg_mask = ndvi > 0.2

# Percentage of vegetation
veg_percent = (np.sum(veg_mask) / veg_mask.size) * 100

# Plot results
plt.figure(figsize=(12,4))

plt.subplot(1,3,1)
plt.title("Original Image")
plt.imshow(img.astype(np.uint8))
plt.axis("off")

plt.subplot(1,3,2)
plt.title("NDVI Map")
plt.imshow(ndvi, cmap="RdYlGn")
plt.axis("off")

plt.subplot(1,3,3)
plt.title(f"Vegetation {veg_percent:.2f}%")
plt.imshow(veg_mask, cmap="gray")
plt.axis("off")

plt.tight_layout()
plt.show()

print("Vegetation Coverage:", veg_percent)