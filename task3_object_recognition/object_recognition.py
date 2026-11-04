
import cv2
from pathlib import Path
from ultralytics import YOLO

# 1. Project paths
BASE_DIR = Path(__file__).resolve().parents[1]

IMAGE_PATH = BASE_DIR / "datasets" / "original photo.jpeg"

OUTPUT_DIR = BASE_DIR / "outputs" / "task3_object_recognition"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 2. Check image
if not IMAGE_PATH.exists():
    raise FileNotFoundError(f"Image not found: {IMAGE_PATH}")

image = cv2.imread(str(IMAGE_PATH))

if image is None:
    raise ValueError("Unable to read the image.")

print("Image loaded successfully!")

# 3. Load YOLO model
print("Loading YOLO model...")
model = YOLO("yolov8n.pt")

# 4. Perform object detection
print("Performing object recognition...")

results = model.predict(
    source=image,
    conf=0.25,
    verbose=False
)

# 5. Display detected objects
result = results[0]

print("\nOBJECT RECOGNITION RESULTS")
print("--------------------------")

if len(result.boxes) > 0:
    for box in result.boxes:
        class_id = int(box.cls[0])
        confidence = float(box.conf[0])
        object_name = model.names[class_id]

        print(
            f"Object: {object_name}, "
            f"Confidence: {confidence:.2f}"
        )
else:
    print("No objects detected.")

print("\nTotal objects detected:", len(result.boxes))

# 6. Draw bounding boxes
annotated_image = result.plot()

# 7. Save output image
output_path = OUTPUT_DIR / "object_detection_result.jpg"

cv2.imwrite(str(output_path), annotated_image)

print("\nTASK 3 COMPLETED SUCCESSFULLY!")
print("Output saved at:", output_path)

# 8. Display result
cv2.imshow("YOLO Object Recognition", annotated_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
