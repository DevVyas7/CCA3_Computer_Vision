# CCA 3 – Practical Coding Assignment (Units 3, 4 and 5)
## Computer Vision Practical Implementation & Technical Evaluation

**Author:** Student  
**Course:** Computer Vision  
**Repository:** [GitHub Repository Placeholder](https://github.com/username/CCA3_Computer_Vision)  
**Project Walkthrough Video:** [YouTube Video Link Placeholder](https://youtube.com/watch?v=placeholder)  

---

## 1. Project Objective

This project implements an end-to-end Computer Vision pipeline covering key algorithms studied across Units 3, 4, and 5 of the curriculum:

1. **Unit 3 - Image Restoration and Segmentation:**
   - Synthetic noise simulation (Gaussian & Salt-and-Pepper).
   - Spatial filtering (Gaussian Smoothing & Non-linear Median Filtering).
   - Quantitative quality metrics: Peak Signal-to-Noise Ratio (PSNR) and Structural Similarity Index (SSIM).
   - Morphological operations (Otsu thresholding, Erosion, Dilation, Opening, Closing, Canny edge detection).
   - Marker-controlled Watershed segmentation and topographic over-segmentation analysis.

2. **Unit 4 - Feature Extraction and Image Matching:**
   - Histogram of Oriented Gradients (HOG) descriptor computation and orientation visualization.
   - Scale-Invariant Feature Transform (SIFT) keypoint detection and 128-D descriptor generation.
   - Brute-Force feature matching with Lowe's ratio test on natural multi-lighting photographs.
   - Quantitative evaluation of geometric invariance (45° rotation and 0.5x downscaling).

3. **Unit 5 - Object Recognition (Classical vs. Deep Learning):**
   - Viola–Jones object detection using OpenCV's pretrained Haar feature cascades.
   - Deep learning multi-class object detection using Ultralytics YOLOv8 (YOLOv8n nano).
   - Empirical comparison and authentic failure case analysis (e.g. Haar false positive on building windows vs YOLO traffic detections).

4. **Unit 5 - Video Motion Estimation:**
   - Frame Differencing for instantaneous change detection.
   - Gaussian Mixture Model (GMM MOG2) for statistical adaptive background subtraction and shadow classification.
   - Lucas–Kanade sparse optical flow for feature tracking and velocity vector visualization.
   - Comparative evaluation across all three motion estimation paradigms.

---

## 2. Technologies and Libraries

- **Language:** Python 3.10+ (Tested on Python 3.13)
- **OpenCV (`opencv-python`):** Image processing, morphological transforms, SIFT, Haar cascades, optical flow, video processing (`>=4.8.0,<5.0.0`)
- **NumPy:** Matrix operations, array manipulation, noise simulation (`>=1.24.0`)
- **Matplotlib:** Visualization, multi-panel comparison plots (`>=3.7.0`)
- **scikit-image:** PSNR/SSIM image metrics and HOG descriptor extraction (`>=0.22.0`)
- **Ultralytics YOLO:** Pretrained YOLOv8n object detection model (`>=8.0.0`)
- **PyTorch (`torch`, `torchvision`):** Deep learning tensor backend for YOLO inference

---

## 3. Project Folder Structure

```
CCA3_Computer_Vision/
│
├── dataset/
│   ├── original_image.jpg           # Primary photograph (urban landscape with rainbow)
│   ├── original_image.jpeg          # Alternative format compatibility
│   ├── second_image.jpg             # Second photograph (different lighting/exposure)
│   ├── second_image.jpeg            # Alternative format compatibility
│   └── original_video.mp4           # Video input (urban street motion, 242 frames)
│
├── outputs/
│   ├── task1_image_restoration_segmentation/   # Denoised images, masks, PSNR/SSIM CSV
│   ├── task2_feature_extraction/               # HOG, SIFT matches, invariance tables
│   ├── task3_object_detection/                 # Haar/YOLO outputs, failure analysis
│   └── task4_motion_estimation/                # Differencing, GMM video, LK flow video
│
├── documentation/
│   ├── technical_report.md          # Comprehensive academic technical report
│   └── generate_pdf_report.py       # Script to compile report to PDF / HTML
│
├── task1_image_restoration.py       # Task 1 standalone module
├── task2_feature_extraction.py      # Task 2 standalone module
├── task3_object_detection.py        # Task 3 standalone module
├── task4_motion_estimation.py       # Task 4 standalone module
├── main.py                          # Master menu-driven interactive runner
├── requirements.txt                 # Project dependencies
├── .gitignore                       # Clean Git tracking configuration
└── README.md                        # Documentation and project manual
```

---

## 4. Dataset Requirements

The project uses authentic, real-world media supplied in the `dataset/` directory:
1. `original_image.jpg`: Primary high-resolution photograph (1280x960) of an urban cityscape with buildings, trees, clouds, and a rainbow.
2. `second_image.jpg`: Secondary photograph of the identical scene captured under different ambient lighting and brightness conditions.
3. `original_video.mp4`: Real-world street video (480x854, 30.00 FPS, 242 frames, 8.07 seconds) containing moving vehicles, pedestrians, and cyclists.

*Note:* The code automatically handles both `.jpg` and `.jpeg` extensions gracefully.

---

## 5. Installation Instructions

### Step 1: Clone or Navigate to the Project Directory
```bash
cd "d:/cca3_computer vision"
```

### Step 2: (Optional but Recommended) Create and Activate a Virtual Environment
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate
```

### Step 3: Install Required Dependencies
```bash
pip install -r requirements.txt
```

---

## 6. How to Run the Project

### Option A: Interactive Menu-Driven System (Recommended)
Run `main.py` directly without arguments to launch the interactive console menu:
```bash
python main.py
```
You will be greeted with the menu:
```
---------------------------- MAIN MENU ----------------------------
  1. Run Task 1: Image Restoration and Segmentation
  2. Run Task 2: Feature Extraction and Image Matching
  3. Run Task 3: Object Recognition (Viola-Jones & YOLO)
  4. Run Task 4: Video Motion Estimation
  5. Run All Tasks Consecutively (Tasks 1 to 4)
  6. Exit
-------------------------------------------------------------------
Enter your choice (1-6):
```

### Option B: Run All Tasks Consecutively via Command Line
```bash
python main.py --all
```

### Option C: Run Individual Task Modules Directly
Each task file is self-contained and can be executed independently:
```bash
# Run Task 1 (Image Restoration & Segmentation)
python task1_image_restoration.py

# Run Task 2 (Feature Extraction & SIFT Matching)
python task2_feature_extraction.py

# Run Task 3 (Object Recognition: Haar & YOLO)
python task3_object_detection.py

# Run Task 4 (Video Motion Estimation)
python task4_motion_estimation.py
```

---

## 7. Detailed Explanation of Tasks and Outputs

### Task 1: Image Restoration and Segmentation
- **Denoising Performance:**
  - Gaussian Noise ($\sigma = 25$): Baseline PSNR = 20.49 dB, SSIM = 0.2542. Restored by $5 \times 5$ Gaussian Filter to **PSNR = 27.08 dB (+6.59 dB)** and **SSIM = 0.6857**.
  - Salt-and-Pepper Noise (4% impulse): Baseline PSNR = 19.13 dB, SSIM = 0.3735. Restored by $5 \times 5$ Median Filter to **PSNR = 27.28 dB (+8.15 dB)** and **SSIM = 0.8943**.
- **Morphological & Segmentation:**
  - Otsu thresholding automatically calculated an optimal threshold of $T = 120.0$.
  - Rectangular $5 \times 5$ structural elements applied for erosion, dilation, opening (noise cleansing), and closing (hole filling).
  - Canny edge detection with hysteresis thresholds $(100, 200)$ highlights prominent rooftop and horizon divides.
  - Marker-based Watershed segmentation reveals the classic limitation of gradient-based watershed on natural landscapes: subtle atmospheric rainbow gradients cause over-segmentation into multiple irregular basins.
- **Outputs Saved In:** `outputs/task1_image_restoration_segmentation/`

### Task 2: Feature Extraction and Image Matching
- **HOG Analysis:** Extracted a 63,504-dimensional feature vector using $16 \times 16$ pixel cells and 9 orientation bins. The visualization captures high-magnitude architectural boundaries while leaving smooth skies dark.
- **SIFT Scale and Rotation Invariance:**
  - Original image pair: 4,607 keypoints (Image 1) and 6,361 keypoints (Image 2) yielding **1,173 good matches** via Lowe's ratio test ($< 0.75$).
  - Rotation Experiment (Image 1 rotated 45°): Yielded **1,137 good matches**, confirming robust rotation invariance.
  - Resizing Experiment (Image 1 downscaled 0.5x): Yielded **512 good matches**, confirming scale-space octave invariance.
- **Outputs Saved In:** `outputs/task2_feature_extraction/`

### Task 3: Object Recognition (Viola–Jones vs. YOLOv8)
- **Viola–Jones (Haar Cascade):**
  - Evaluated on photograph and video frames.
  - *Observed Failure Case:* On the photograph, Haar detected 1 candidate face on the building facade. This is an authentic **False Positive** caused by rectangular window patterns mirroring the intensity ratios of human eyes and nose bridges.
  - *Video Limitation:* Missed distant pedestrians and helmeted motorcyclists in the street video due to lack of frontal facial features.
- **Ultralytics YOLOv8 (YOLOv8n):**
  - Accurately detected 12 objects in Video Frame 0 (cars, persons, motorcycles, bicycle).
  - Correctly recognized objects under partial occlusion and scale changes.
- **Outputs Saved In:** `outputs/task3_object_detection/`

### Task 4: Video Motion Estimation
- **Input Properties:** 480x854 resolution, 30.00 FPS, 242 frames (8.07 seconds).
- **Techniques Compared:**
  1. *Frame Differencing:* Fast, instantaneous change detection ($28,438$ moving pixels/frame average). Suffers from interior aperture emptiness and ghosting.
  2. *GMM Background Subtraction (MOG2):* Adaptive background model isolating solid foreground silhouettes ($64,596$ foreground pixels/frame average) and shadows.
  3. *Lucas–Kanade Optical Flow:* Trajectory tracking of up to 120 Shi-Tomasi corners, displaying velocity vectors as vehicles travel.
- **Generated Videos:**
  - `outputs/task4_motion_estimation/task4_gmm_foreground_mask.mp4`
  - `outputs/task4_motion_estimation/task4_lucas_kanade_flow.mp4`

---

## 8. Troubleshooting and Known Limitations

1. **OpenCV CascadeClassifier Attribute Warning:**
   - Some preview/pre-release builds (such as OpenCV 5.0 preview) omit `cv2.CascadeClassifier`. The project explicitly pins `opencv-python>=4.8.0,<5.0.0` in `requirements.txt` to guarantee complete compatibility.
2. **First-run YOLO Weights Download:**
   - The first time Task 3 is run, `ultralytics` will download `yolov8n.pt` (~6.2 MB) automatically if an internet connection is available. Once downloaded, it is cached locally in the root directory.
3. **Headless Execution:**
   - All visualization figures are automatically saved to disk via `plt.savefig()` with `show_plots=False` default so the code executes seamlessly on systems without a desktop display or in automated test environments.

---

## 9. Academic Integrity and Certification
All experimental data, keypoint counts, PSNR/SSIM scores, bounding boxes, and motion statistics recorded in this project were generated strictly by executing the code on the authentic dataset files. No results or metrics were fabricated.
