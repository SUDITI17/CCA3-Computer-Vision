
import cv2
import numpy as np
from pathlib import Path

# ==========================================
# 1. PROJECT PATHS
# ==========================================

BASE_DIR = Path(__file__).resolve().parents[1]

VIDEO_PATH = BASE_DIR / "datasets" / "original_video.mp4"

OUTPUT_DIR = BASE_DIR / "outputs" / "task4_motion_estimation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ==========================================
# 2. OPEN VIDEO
# ==========================================

cap = cv2.VideoCapture(str(VIDEO_PATH))

if not cap.isOpened():
    raise FileNotFoundError(
        f"Video not found: {VIDEO_PATH}"
    )

print("Video loaded successfully!")

ret, previous_frame = cap.read()

if not ret:
    raise ValueError("Unable to read video!")

height, width = previous_frame.shape[:2]

fps = cap.get(cv2.CAP_PROP_FPS)

if fps <= 0:
    fps = 20

# Create output MP4 video
fourcc = cv2.VideoWriter_fourcc(*"mp4v")

video_writer = cv2.VideoWriter(
    str(OUTPUT_DIR / "motion_output.mp4"),
    fourcc,
    fps,
    (width, height)
)

if not video_writer.isOpened():
    cap.release()
    raise RuntimeError("Unable to create output video!")

previous_gray = cv2.cvtColor(
    previous_frame, cv2.COLOR_BGR2GRAY
)

# Save first frame
cv2.imwrite(
    str(OUTPUT_DIR / "first_frame.jpg"),
    previous_frame
)

# Include first frame in output video
video_writer.write(previous_frame)


# ==========================================
# 3. GMM BACKGROUND SUBTRACTION
# ==========================================

background_subtractor = cv2.createBackgroundSubtractorMOG2(
    history=500,
    varThreshold=25,
    detectShadows=False
)

background_subtractor.apply(previous_frame)


# ==========================================
# 4. LUCAS-KANADE OPTICAL FLOW
# ==========================================

feature_params = dict(
    maxCorners=100,
    qualityLevel=0.3,
    minDistance=7,
    blockSize=7
)

lk_params = dict(
    winSize=(15, 15),
    maxLevel=2,
    criteria=(
        cv2.TERM_CRITERIA_EPS |
        cv2.TERM_CRITERIA_COUNT,
        10,
        0.03
    )
)

previous_points = cv2.goodFeaturesToTrack(
    previous_gray,
    mask=None,
    **feature_params
)

if previous_points is None:
    cap.release()
    video_writer.release()
    raise ValueError("No tracking points found!")

print("Initial tracking points:", len(previous_points))


# ==========================================
# 5. PROCESS VIDEO FRAMES
# ==========================================

frame_count = 0
sample_found = False

while True:

    ret, current_frame = cap.read()

    if not ret:
        break

    frame_count += 1

    current_gray = cv2.cvtColor(
        current_frame, cv2.COLOR_BGR2GRAY
    )

    # A. FRAME DIFFERENCING

    difference = cv2.absdiff(
        previous_gray,
        current_gray
    )

    _, frame_mask = cv2.threshold(
        difference,
        25,
        255,
        cv2.THRESH_BINARY
    )

    # B. GMM BACKGROUND SUBTRACTION

    foreground_mask = background_subtractor.apply(
        current_frame
    )

    # C. LUCAS-KANADE OPTICAL FLOW

    current_points, status, error = cv2.calcOpticalFlowPyrLK(
        previous_gray,
        current_gray,
        previous_points,
        None,
        **lk_params
    )

    motion_image = current_frame.copy()

    good_count = 0

    if current_points is not None and status is not None:

        good_new = current_points[status.flatten() == 1]
        good_old = previous_points[status.flatten() == 1]

        good_count = len(good_new)

        for new, old in zip(good_new, good_old):

            x_new, y_new = new.ravel()
            x_old, y_old = old.ravel()

            cv2.arrowedLine(
                motion_image,
                (int(x_old), int(y_old)),
                (int(x_new), int(y_new)),
                (0, 255, 0),
                2
            )

            cv2.circle(
                motion_image,
                (int(x_new), int(y_new)),
                4,
                (0, 0, 255),
                -1
            )

        if good_count > 0:
            previous_points = good_new.reshape(-1, 1, 2)

    # Refresh tracking points if needed

    if good_count < 5:

        previous_points = cv2.goodFeaturesToTrack(
            current_gray,
            mask=None,
            **feature_params
        )

        if previous_points is None:
            previous_points = np.empty(
                (0, 1, 2), dtype=np.float32
            )

    # D. WRITE PROCESSED FRAME TO MP4

    video_writer.write(motion_image)

    # ======================================
    # 6. SAVE SAMPLE OUTPUTS AT FRAME 30
    # ======================================

    if frame_count == 30:

        sample_found = True

        cv2.imwrite(
            str(OUTPUT_DIR / "second_frame.jpg"),
            current_frame
        )

        cv2.imwrite(
            str(OUTPUT_DIR / "frame_differencing.jpg"),
            frame_mask
        )

        cv2.imwrite(
            str(OUTPUT_DIR / "gmm_background_subtraction.jpg"),
            foreground_mask
        )

        cv2.imwrite(
            str(OUTPUT_DIR / "lucas_kanade_optical_flow.jpg"),
            motion_image
        )

        # Create comparison image

        comparison = np.hstack((
            cv2.cvtColor(frame_mask, cv2.COLOR_GRAY2BGR),
            cv2.cvtColor(foreground_mask, cv2.COLOR_GRAY2BGR),
            motion_image
        ))

        cv2.imwrite(
            str(OUTPUT_DIR / "motion_comparison.jpg"),
            comparison
        )

        print("\nMOTION ESTIMATION RESULTS")
        print("--------------------------------")
        print("Frame number:", frame_count)
        print(
            "Frame difference pixels:",
            cv2.countNonZero(frame_mask)
        )
        print(
            "GMM foreground pixels:",
            cv2.countNonZero(foreground_mask)
        )
        print(
            "Lucas-Kanade tracked points:",
            good_count
        )

    previous_gray = current_gray


# ==========================================
# 7. RELEASE RESOURCES
# ==========================================

cap.release()
video_writer.release()

print("\nTotal video frames processed:", frame_count)

if sample_found:

    print("TASK 4 COMPLETED SUCCESSFULLY!")

    print("MP4 video saved:",
          OUTPUT_DIR / "motion_output.mp4")

    print("All outputs saved in:", OUTPUT_DIR)

else:

    print("Video is too short for frame 30.")

    print("Try a shorter sample frame index.")


cv2.destroyAllWindows()
