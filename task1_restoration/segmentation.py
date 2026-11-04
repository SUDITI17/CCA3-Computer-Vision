
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# 1. Set paths
BASE_DIR = Path(__file__).resolve().parents[1]

IMAGE_PATH = BASE_DIR / "datasets" / "original photo.jpeg"

OUTPUT_DIR = BASE_DIR / "outputs" / "task1_segmentation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 2. Read original image
image = cv2.imread(str(IMAGE_PATH))

if image is None:
    raise FileNotFoundError(f"Image not found: {IMAGE_PATH}")

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 3. Thresholding
_, binary = cv2.threshold(
    gray, 0, 255,
    cv2.THRESH_BINARY + cv2.THRESH_OTSU
)

# 4. Create morphological kernel
kernel = np.ones((5, 5), np.uint8)

# 5. Erosion
erosion = cv2.erode(binary, kernel, iterations=1)

# 6. Dilation
dilation = cv2.dilate(binary, kernel, iterations=1)

# 7. Opening
opening = cv2.morphologyEx(
    binary, cv2.MORPH_OPEN, kernel
)

# 8. Closing
closing = cv2.morphologyEx(
    binary, cv2.MORPH_CLOSE, kernel
)

# 9. Canny Edge Detection
edges = cv2.Canny(gray, 100, 200)

# 10. Watershed Segmentation
# Use Otsu's binary image as the foreground mask
sure_bg = cv2.dilate(
    binary, kernel, iterations=3
)

distance = cv2.distanceTransform(
    binary, cv2.DIST_L2, 5
)

_, sure_fg = cv2.threshold(
    distance,
    0.4 * distance.max(),
    255,
    cv2.THRESH_BINARY
)

sure_fg = np.uint8(sure_fg)

unknown = cv2.subtract(sure_bg, sure_fg)

_, markers = cv2.connectedComponents(sure_fg)

markers = markers + 1
markers[unknown == 255] = 0

watershed_result = image.copy()

cv2.watershed(watershed_result, markers)

watershed_result[markers == -1] = [0, 0, 255]

# 11. Collect all results
results = {
    "original": image,
    "grayscale": gray,
    "binary_threshold": binary,
    "erosion": erosion,
    "dilation": dilation,
    "opening": opening,
    "closing": closing,
    "canny_edges": edges,
    "watershed": watershed_result
}

# 12. Save all results
for name, result in results.items():
    cv2.imwrite(
        str(OUTPUT_DIR / f"{name}.jpg"),
        result
    )

# 13. Display results
plt.figure(figsize=(14, 12))

for index, (name, result) in enumerate(results.items()):
    plt.subplot(3, 3, index + 1)

    if len(result.shape) == 3:
        result = cv2.cvtColor(
            result, cv2.COLOR_BGR2RGB
        )

    plt.imshow(result, cmap="gray")
    plt.title(name.replace("_", " ").title())
    plt.axis("off")

plt.tight_layout()

plt.savefig(
    str(OUTPUT_DIR / "segmentation_comparison.png")
)

plt.show()

print("SEGMENTATION COMPLETED SUCCESSFULLY!")
print("All output images saved in:", OUTPUT_DIR)
