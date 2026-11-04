
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import hog
from pathlib import Path

# 1. Project paths
BASE_DIR = Path(__file__).resolve().parents[1]

IMAGE1_PATH = BASE_DIR / "datasets" / "original photo.jpeg"
IMAGE2_PATH = BASE_DIR / "datasets" / "second image.jpeg"
OUTPUT_DIR = BASE_DIR / "outputs" / "task2_feature_extraction"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 2. Read photographs
img1 = cv2.imread(str(IMAGE1_PATH))
img2 = cv2.imread(str(IMAGE2_PATH))

if img1 is None:
    raise FileNotFoundError(f"First image not found: {IMAGE1_PATH}")

if img2 is None:
    raise FileNotFoundError(f"Second image not found: {IMAGE2_PATH}")

gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# 3. HOG Feature Extraction
resized_gray = cv2.resize(gray1, (128, 128))

hog_features, hog_image = hog(
    resized_gray,
    orientations=9,
    pixels_per_cell=(8, 8),
    cells_per_block=(2, 2),
    visualize=True
)

print("HOG Feature Extraction Completed!")
print("Number of HOG features:", len(hog_features))

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(gray1, cmap="gray")
plt.title("Original Grayscale")

plt.subplot(1, 2, 2)
plt.imshow(hog_image, cmap="gray")
plt.title("HOG Visualization")

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "hog_visualization.png")
plt.show()

# 4. SIFT Keypoint Detection
sift = cv2.SIFT_create()

kp1, des1 = sift.detectAndCompute(gray1, None)
kp2, des2 = sift.detectAndCompute(gray2, None)

print("\nSIFT Keypoint Detection")
print("Image 1 keypoints:", len(kp1))
print("Image 2 keypoints:", len(kp2))

keypoints1 = cv2.drawKeypoints(
    img1, kp1, None,
    flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
)

keypoints2 = cv2.drawKeypoints(
    img2, kp2, None,
    flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
)

cv2.imwrite(str(OUTPUT_DIR / "sift_image1.jpg"), keypoints1)
cv2.imwrite(str(OUTPUT_DIR / "sift_image2.jpg"), keypoints2)

# 5. Create feature matching function
def match_images(image_a, image_b, name):

    gray_a = cv2.cvtColor(image_a, cv2.COLOR_BGR2GRAY)
    gray_b = cv2.cvtColor(image_b, cv2.COLOR_BGR2GRAY)

    key_a, desc_a = sift.detectAndCompute(gray_a, None)
    key_b, desc_b = sift.detectAndCompute(gray_b, None)

    if desc_a is None or desc_b is None:
        print(name, ": Not enough features for matching")
        return 0

    matcher = cv2.BFMatcher(cv2.NORM_L2)

    pairs = matcher.knnMatch(desc_a, desc_b, k=2)

    good_matches = []

    for pair in pairs:
        if len(pair) == 2:
            m, n = pair

            if m.distance < 0.75 * n.distance:
                good_matches.append(m)

    result = cv2.drawMatches(
        image_a, key_a,
        image_b, key_b,
        good_matches[:50],
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )

    cv2.imwrite(
        str(OUTPUT_DIR / f"{name}.jpg"),
        result
    )

    print(name, "Good matches:", len(good_matches))

    return len(good_matches)


# 6. Original Image Matching
original_matches = match_images(
    img1, img2, "original_matching"
)

# 7. Rotation Experiment
height, width = img2.shape[:2]

matrix = cv2.getRotationMatrix2D(
    (width // 2, height // 2),
    30,
    1.0
)

rotated_image = cv2.warpAffine(
    img2, matrix, (width, height)
)

cv2.imwrite(
    str(OUTPUT_DIR / "rotated_image.jpg"),
    rotated_image
)

rotated_matches = match_images(
    img1, rotated_image, "rotated_matching"
)

# 8. Resizing Experiment
resized_image = cv2.resize(
    img2, None,
    fx=0.75,
    fy=0.75
)

cv2.imwrite(
    str(OUTPUT_DIR / "resized_image.jpg"),
    resized_image
)

resized_matches = match_images(
    img1, resized_image, "resized_matching"
)

# 9. Comparison of results
print("\nFEATURE MATCHING COMPARISON")
print("--------------------------------")

print("Original matches:", original_matches)
print("Rotated matches:", rotated_matches)
print("Resized matches:", resized_matches)

# 10. Save comparison graph
labels = ["Original", "Rotated", "Resized"]

values = [
    original_matches,
    rotated_matches,
    resized_matches
]

plt.figure(figsize=(8, 5))

plt.bar(labels, values)

plt.title("SIFT Feature Matching Comparison")
plt.xlabel("Image Transformation")
plt.ylabel("Number of Good Matches")

plt.savefig(
    OUTPUT_DIR / "matching_comparison.png"
)

plt.show()

print("\nTASK 2 COMPLETED SUCCESSFULLY!")
print("All results saved in:", OUTPUT_DIR)
