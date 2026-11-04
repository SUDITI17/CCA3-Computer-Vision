# CCA 3 – Computer Vision and Image Processing

## Project Overview

This project is developed as part of the CCA 3 Practical Coding Assignment covering Units 3, 4 and 5.

The project implements image restoration, image segmentation, feature extraction, object recognition and video motion estimation using Python and computer vision libraries.

The assignment uses an original photograph and a self-recorded video as input datasets.

## Objectives

The main objectives of this project are:

- Implement image restoration techniques to remove noise.
- Perform morphological processing and image segmentation.
- Extract and match image features using HOG and SIFT.
- Detect objects using Haar Cascade and YOLO.
- Estimate video motion using frame differencing, GMM and Lucas-Kanade optical flow.
- Compare the results of different computer vision techniques.

## Technologies Used

| Technology | Purpose |
|---|---|
| Python | Programming language |
| OpenCV | Image and video processing |
| NumPy | Numerical computations |
| Matplotlib | Image visualization |
| scikit-image | HOG, PSNR and SSIM |
| Ultralytics YOLO | Object detection |
| VS Code | Development environment |

## Project Tasks

### Task 1: Image Restoration and Segmentation

**Image Restoration**
- Gaussian noise addition.
- Salt-and-pepper noise addition.
- Median filtering.
- Gaussian filtering.
- PSNR and SSIM evaluation.

**Image Segmentation**
- Binary thresholding.
- Erosion and dilation.
- Opening and closing.
- Canny edge detection.
- Watershed segmentation.

### Task 2: Feature Extraction and Image Matching

- Histogram of Oriented Gradients (HOG).
- SIFT keypoint detection.
- SIFT descriptor extraction.
- Feature matching between two images.
- Matching after image rotation.
- Matching after image resizing.

### Task 3: Object Recognition

- Face detection using Haar Cascade.
- Object detection using pretrained YOLO.
- Bounding box visualization.
- Class labels and confidence scores.
- Comparison of detection results.

### Task 4: Video Motion Estimation

- Video frame extraction.
- Frame differencing.
- GMM-based background subtraction.
- Lucas-Kanade optical flow.
- Motion vector visualization.
- Comparison of motion estimation techniques.

## Project Folder Structure

```text
CCA3_computer_vision/
│
├── datasets/
│   ├── original_image.jpg
│   ├── second_image.jpg
│   └── original_video.mp4
│
├── outputs/
│   ├── task1_restoration/
│   ├── task1_segmentation/
│   ├── task2_feature_extraction/
│   ├── task3_object_recognition/
│   └── task4_motion_estimation/
│
├── task1_restoration/
│   └── restoration.py
│
├── task1_segmentation/
│   └── segmentation.py
│
├── task2_feature_extraction/
│   └── feature_extraction.py
│
├── task3_object_recognition/
│   └── object_recognition.py
│
├── task4_motion_estimation/
│   └── motion_estimation.py
│
├── documentation/
├── yolov8n.pt
├── README.md
├── requirements.txt
└── .gitignore
```

*Note: The folder structure above describes the intended organization. Keep the actual file and folder names consistent with your VS Code project.*

## Installation and Setup

### 1. Clone the Repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

### 2. Open the Project Folder

```bash
cd CCA3_computer_vision
```

### 3. Create a Virtual Environment

```bash
python -m venv .venv
```

### 4. Activate the Virtual Environment

For Windows PowerShell:

```bash
.venv\Scripts\Activate.ps1
```

### 5. Install Required Libraries

```bash
pip install -r requirements.txt
```

## Running the Project

Run the Python scripts from the main project directory.

```bash
python task1_restoration/restoration.py
```

```bash
python task1_segmentation/segmentation.py
```

```bash
python task2_feature_extraction/feature_extraction.py
```

```bash
python task3_object_recognition/object_recognition.py
```

```bash
python task4_motion_estimation/motion_estimation.py
```

The scripts generate their respective output images and results inside the outputs directory.

## Dataset Information

The project uses:

- An original photograph captured using a mobile phone or camera.
- A second photograph of the same object or scene.
- An original self-recorded video of approximately 15–30 seconds.

The original input files are preserved in the datasets folder.

Large video files can be obtained from the Google Drive submission folder.

## Output Results

The project generates:

- Noisy and restored images.
- Binary masks and segmentation results.
- HOG visualizations and SIFT matching results.
- Haar Cascade and YOLO detection images.
- Motion estimation images and a processed MP4 video.

## Conclusion

This project demonstrates the implementation of important computer vision techniques using Python. Different image processing, feature extraction, object detection and motion estimation algorithms are implemented and compared through practical experiments.

## Author

**Name:** Suditi Kumari

**PRN:** 1272251102

**Course:** Master of Computer Applications (MCA)

**Assignment:** CCA 3 – Computer Vision and Image Processing

## GitHub Repository

Add your GitHub repository URL here:

YOUR_GITHUB_REPOSITORY_URL
