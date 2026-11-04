
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from pathlib import Path

# 1. Set project paths
BASE_DIR = Path(__file__).resolve().parents[1]
IMAGE_PATH = BASE_DIR / "datasets" / "original photo.jpeg"
OUTPUT_DIR = BASE_DIR / "outputs" / "task1_restoration"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 2. Read original photograph
original = cv2.imread(str(IMAGE_PATH))

if original is None:
    raise FileNotFoundError(f"Image not found: {IMAGE_PATH}")

# 3. Add Gaussian noise
np.random.seed(42)

gaussian_noise = np.random.normal(
    0, 25, original.shape
).astype(np.float32)

gaussian_noisy = np.clip(
    original.astype(np.float32) + gaussian_noise,
    0, 255
).astype(np.uint8)

# 4. Add Salt-and-Pepper noise
salt_pepper = original.copy()
noise_amount = 0.04

random_values = np.random.random(original.shape[:2])

salt_pepper[random_values < noise_amount / 2] = 255
salt_pepper[random_values > 1 - noise_amount / 2] = 0

# 5. Apply Median Filtering
median_gaussian = cv2.medianBlur(gaussian_noisy, 5)
median_sp = cv2.medianBlur(salt_pepper, 5)

# 6. Apply Gaussian Filtering
gaussian_filtered = cv2.GaussianBlur(
    gaussian_noisy, (5, 5), 1
)

gaussian_filtered_sp = cv2.GaussianBlur(
    salt_pepper, (5, 5), 1
)

# 7. Calculate PSNR and SSIM
images = {
    "Original": original,
    "Gaussian Noise": gaussian_noisy,
    "Salt Pepper Noise": salt_pepper,
    "Median Gaussian": median_gaussian,
    "Gaussian Filter": gaussian_filtered,
    "Median Salt Pepper": median_sp,
    "Gaussian Filter SP": gaussian_filtered_sp
}

print("\nIMAGE RESTORATION RESULTS")
print("-" * 65)
print(f"{'Image':<25} {'PSNR':<18} {'SSIM':<15}")
print("-" * 65)

for name, image in images.items():

    if name == "Original":
        continue

    if "Gaussian" in name and name != "Gaussian Filter SP":
        reference = original
    else:
        reference = original

    psnr = peak_signal_noise_ratio(
        reference, image, data_range=255
    )

    ssim = structural_similarity(
        reference,
        image,
        channel_axis=-1,
        data_range=255
    )

    print(f"{name:<25} {psnr:<18.2f} {ssim:<15.4f}")

# 8. Save all output images
for name, image in images.items():
    filename = name.lower().replace(" ", "_") + ".jpg"
    cv2.imwrite(str(OUTPUT_DIR / filename), image)

# 9. Display comparison
plt.figure(figsize=(14, 10))

for index, (name, image) in enumerate(images.items()):
    plt.subplot(3, 3, index + 1)

    rgb_image = cv2.cvtColor(
        image, cv2.COLOR_BGR2RGB
    )

    plt.imshow(rgb_image)
    plt.title(name)
    plt.axis("off")

plt.tight_layout()

plt.savefig(
    str(OUTPUT_DIR / "restoration_comparison.png")
)

plt.show()

print("\nAll images saved successfully!")
print("Output folder:", OUTPUT_DIR)
