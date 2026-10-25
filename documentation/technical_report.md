# CCA 3 – Practical Coding Assignment (Units 3, 4 and 5)
## Comprehensive Technical Report on Image Restoration, Segmentation, Feature Extraction, Object Recognition, and Video Motion Estimation

---

### **Student Name:** [Student Name Placeholder]  
### **Student Roll / ID:** [Student ID Placeholder]  
### **Course Title:** Computer Vision (CCA 3)  
### **Institution:** [College / University Department of Computer Science]  
### **Date of Submission:** October 2026  
### **Project Repository:** [GitHub Repository Placeholder](https://github.com/username/CCA3_Computer_Vision)  
### **Project Video Walkthrough:** [YouTube Video Link Placeholder](https://youtube.com/watch?v=placeholder)  

---

## Executive Summary

This report documents the design, mathematical foundations, implementation, and empirical evaluation of a complete Computer Vision pipeline covering Units 3, 4, and 5 of the Computer Vision curriculum. Using an authentic dataset comprising multi-exposure urban photographs of a city skyline with a rainbow and an urban traffic video, the following four primary core areas were systematically investigated:

1. **Image Restoration & Segmentation (Unit 3):** Modeling of additive Gaussian thermal noise and impulse salt-and-pepper transmission noise; spatial domain restoration via Gaussian and non-linear median filtering; objective quantitative validation via Peak Signal-to-Noise Ratio (PSNR) and Structural Similarity Index (SSIM); binary Otsu thresholding; morphological transforms (erosion, dilation, opening, closing); Canny edge detection; and marker-controlled watershed segmentation with an evaluation of over-segmentation in natural outdoor environments.
2. **Feature Extraction & Image Matching (Unit 4):** Spatial gradient orientation analysis using Histogram of Oriented Gradients (HOG); keypoint detection and 128-dimensional descriptor generation using the Scale-Invariant Feature Transform (SIFT); Brute-Force feature correspondence with Lowe's second-nearest-neighbor ratio test across varying illumination conditions; and rigorous testing of geometric transformation invariance under programmatic 45° rotation and 0.5x downscaling.
3. **Object Recognition: Classical vs. Deep Learning (Unit 5):** Comparison between classical cascaded classifiers (Viola–Jones with integral images and Adaboost) and state-of-the-art deep convolutional networks (Ultralytics YOLOv8 nano); analysis of computational complexity; and documentation of authentic failure modes (including Haar cascade architectural false positives on building window lattices and missed occluded/distant pedestrians).
4. **Video Motion Estimation (Unit 5):** Temporal motion analysis using Frame Differencing, adaptive Gaussian Mixture Model (GMM MOG2) background subtraction with shadow identification, and sparse pyramidal Lucas–Kanade optical flow point tracking.

All measurements, keypoint tallies, and performance metrics recorded herein were obtained through real-time program execution on the provided media files.

---

## 1. Introduction and Objectives

Computer Vision has evolved from classical hand-crafted heuristics and statistical filtering to sophisticated deep convolutional representations. The primary pedagogical objective of this assignment is to bridge theoretical knowledge with practical engineering implementation across three foundational pillars:
- **Image Enhancement and Low-Level Segmentation:** Understanding how digital filters operate on discrete pixel spatial grids to invert corruption processes, and how gradient topography separates object boundaries.
- **Local Invariant Descriptors:** Understanding how gradient histograms and scale-space Difference-of-Gaussians (DoG) representations provide robust geometric and photometric invariance for image matching.
- **High-Level Visual Perception and Dynamics:** Contrasting sliding-window Adaboost cascades with single-shot deep neural detectors, and quantifying motion dynamics in continuous temporal video streams.

---

## 2. Input Dataset Description

The experiments were executed using authentic media depicting real-world scenarios:
1. **Primary Photograph (`dataset/original_image.jpg`):** A high-resolution RGB image ($960 \times 1280$ pixels, 3 channels) capturing an urban panorama consisting of multi-story buildings, foreground trees, sky, cloud layers, and a faint atmospheric rainbow under normal daylight exposure.
2. **Secondary Photograph (`dataset/second_image.jpg`):** An identical scene captured from the same viewpoint under distinct ambient illumination, cloud distribution, and camera exposure conditions, providing a benchmark for photometric invariance.
3. **Street Video (`dataset/original_video.mp4`):** An 8.07-second video recording ($480 \times 854$ resolution, 30.00 FPS, 242 total frames) depicting an urban roadway with moving motor vehicles, motorcycles, bicycles, and pedestrians.

---

## 3. Software Environment and System Configuration

- **Core Runtime:** Python 3.13 / Windows 11 AMD64
- **Computer Vision Framework:** OpenCV (`opencv-python` 4.14.0)
- **Scientific Computing:** NumPy 2.4.2
- **Data Visualization:** Matplotlib 3.10.9
- **Image Science & Metrics:** scikit-image 0.26.0 (`skimage.metrics`, `skimage.feature.hog`)
- **Deep Learning Inference:** Ultralytics YOLOv8 8.4.189 powered by PyTorch 2.11.0
- **Document Compilation:** ReportLab 5.1.0

---

## 4. Task 1: Image Restoration and Segmentation

### 4.1 Theoretical Foundation & Methodology

#### 4.1.1 Noise Modeling
- **Additive Gaussian Noise:** Models electronic amplifier and thermal sensor fluctuations where pixel value $I(x, y)$ is perturbed as:
  $$I_{\text{noisy}}(x, y) = \text{clip}\left(I(x, y) + \mathcal{N}(\mu=0, \sigma=25), 0, 255\right)$$
- **Salt-and-Pepper (Impulse) Noise:** Models transmission errors and bit-drop flaws, where pixels are randomly corrupted with probability $p = 0.04$ to either maximum intensity ($255$, salt) or minimum intensity ($0$, pepper).

#### 4.1.2 Restoration Filters
- **Gaussian Linear Smoothing:** Computes a weighted local average using a 2D Gaussian kernel:
  $$G(x, y) = \frac{1}{2\pi\sigma^2} e^{-\frac{x^2+y^2}{2\sigma^2}}$$
  Applied with kernel dimensions $5 \times 5$ and $\sigma = 1.2$.
- **Median Non-Linear Filtering:** Replaces each central pixel with the median intensity of its spatial neighborhood ($5 \times 5$ window). Because median filtering does not compute an arithmetic average, extreme outliers (0 and 255 impulses) are completely eliminated without blurring sharp edges.

#### 4.1.3 Quantitative Image Quality Metrics
- **Peak Signal-to-Noise Ratio (PSNR):**
  $$\text{PSNR} = 10 \cdot \log_{10}\left(\frac{\text{MAX}_I^2}{\text{MSE}}\right)$$
- **Structural Similarity Index (SSIM):** Compares luminance, contrast, and structural correlation over local windows.

#### 4.1.4 Morphological Operations & Watershed Segmentation
- **Otsu's Global Thresholding:** Automatically determines the threshold $T^*$ that minimizes the intra-class variance of foreground and background pixel distributions.
- **Morphological Operations:** Using a flat $5 \times 5$ structuring element $B$:
  - Erosion: $A \ominus B = \{z \mid (B)_z \subseteq A\}$
  - Dilation: $A \oplus B = \{z \mid (\hat{B})_z \cap A \neq \emptyset\}$
  - Opening: $A \circ B = (A \ominus B) \oplus B$ (clears isolated white foreground speckles)
  - Closing: $A \bullet B = (A \oplus B) \ominus B$ (bridges interior dark cavities)
- **Canny Edge Detection:** Multi-stage algorithm featuring Gaussian smoothing, Sobel gradient computation, non-maximum suppression, and dual-threshold hysteresis ($T_{\text{low}} = 100, T_{\text{high}} = 200$).
- **Marker-Controlled Watershed:** Treats the gradient magnitude as a topographic relief surface where pixel brightness corresponds to altitude. Markers are extracted from the Euclidean distance transform of the binary threshold mask:
  $$D(x, y) = \min_{(x', y') \in \text{Background}} \sqrt{(x - x')^2 + (y - y')^2}$$

---

### 4.2 Empirical Results and Quantitative Data

#### Table 1: Task 1 Quantitative Image Restoration Metrics
| Corrupted Image Input | Filter Applied | PSNR (dB) | SSIM Score | Restoration Assessment |
| :--- | :--- | :---: | :---: | :--- |
| **Gaussian Noisy ($\sigma=25$)** | *None (Raw Noisy)* | 20.49 dB | 0.2542 | Baseline severe degradation |
| **Gaussian Noisy ($\sigma=25$)** | **Gaussian Filter ($5 \times 5$)** | **27.08 dB** | **0.6857** | **Optimal: +6.59 dB gain; smooth noise attenuation** |
| **Gaussian Noisy ($\sigma=25$)** | Median Filter ($5 \times 5$) | 25.95 dB | 0.6617 | Sub-optimal: suppresses noise but distorts continuous gradients |
| **Salt & Pepper (4% Impulse)** | *None (Raw Noisy)* | 19.13 dB | 0.3735 | Baseline severe impulse corruption |
| **Salt & Pepper (4% Impulse)** | **Median Filter ($5 \times 5$)** | **27.28 dB** | **0.8943** | **Optimal: +8.15 dB gain; near-total impulse elimination** |
| **Salt & Pepper (4% Impulse)** | Gaussian Filter ($5 \times 5$) | 26.56 dB | 0.6748 | Defective: smears impulse noise into blurred halos |

#### Observations on Watershed Segmentation
In this natural outdoor scene, Otsu's threshold resolved at $T^* = 120.0$. Applying marker-controlled watershed highlighted both the utility and inherent limitations of intensity-based segmentation:
1. **Building Silhouette Isolation:** The stark radiometric contrast between dark building profiles and the daylight sky resulted in accurate delineation of major architectural divides.
2. **Topographic Over-Segmentation:** Subtle atmospheric gradients within the rainbow arc, variegated cloud density, and complex tree foliage created numerous local minima. Consequently, the watershed algorithm partitioned continuous sky and foliage regions into fragmented polygonal basins. This demonstrates that while watershed excels on isolated convex particles (e.g., biological cells or industrial parts on a clean backdrop), natural outdoor scenes require semantic segmentation (e.g., Mask R-CNN or YOLO-Seg) to separate semantic entities.

---

## 5. Task 2: Feature Extraction and Image Matching

### 5.1 Theoretical Foundation & Methodology

#### 5.1.1 Histogram of Oriented Gradients (HOG)
HOG describes appearance by accumulating localized 1-D histograms of gradient orientations. The input image was gridded into $16 \times 16$ pixel spatial cells with 9 unsigned orientation bins spanning $[0^\circ, 180^\circ]$. Blocks of $2 \times 2$ cells were normalized using the L2-norm to achieve photometric invariance against ambient illumination shifts.

#### 5.1.2 Scale-Invariant Feature Transform (SIFT)
SIFT detects stable feature locations across continuous scale space via Difference-of-Gaussians (DoG):
$$D(x, y, \sigma) = (G(x, y, k\sigma) - G(x, y, \sigma)) * I(x, y)$$
Sub-pixel localization filters low-contrast points and unstable edge responses. Canonical orientations are assigned based on local gradient histograms. Finally, a $4 \times 4$ array of 8-bin orientation histograms generates a 128-dimensional feature vector per keypoint.

#### 5.1.3 Feature Correspondence & Lowe's Ratio Test
Descriptor vectors were matched using the Brute-Force Matcher with Euclidean distance ($L_2$ norm). To eliminate false correspondences from repetitive textures, Lowe's second-nearest-neighbor ratio test was enforced:
$$\frac{\|D_1 - D_{2,\text{best}}\|_2}{\|D_1 - D_{2,\text{second}}\|_2} < 0.75$$

---

### 5.2 Empirical Results and Invariance Experiments

#### Table 2: Task 2 SIFT Keypoint and Matching Evaluation
| Experiment Name | Transformation Applied | Img 1 Keypoints | Img 2 Keypoints | Good Matches ($<0.75$) | Invariance Behavior & Analysis |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **Original Pair (Baseline)** | None (Natural Lighting Shift) | 4,607 | 6,361 | **1,173** | High matching fidelity despite noticeable exposure and cloud differences. |
| **Rotation Invariance** | Programmatic 45° In-Plane Rotation | 4,444 | 6,361 | **1,137** | **96.9% match retention rate**; dominant gradient orientation successfully compensated for rotation. |
| **Scale Invariance** | Downscaling by 50% (0.5x Scale) | 1,423 | 6,361 | **512** | Keypoint count decimation due to lower Nyquist limit; valid correspondences reliably preserved across scales. |

---

## 6. Task 3: Object Recognition (Viola–Jones vs. YOLOv8)

### 6.1 Architecture Comparison

#### 6.1.1 Viola–Jones Detector
- **Feature Representation:** Rectangular Haar-like features (two-, three-, and four-rectangle configurations) computed in constant time $O(1)$ using Integral Images:
  $$ii(x, y) = \sum_{x' \le x, y' \le y} i(x', y')$$
- **Classifier:** Attentional cascade of Adaboost-trained decision stumps that rapidly reject non-face background windows in early stages.
- **Scope:** Specialized single-class detector (human frontal faces).

#### 6.1.2 Ultralytics YOLOv8 (YOLOv8n Nano)
- **Feature Representation:** Deep Convolutional Neural Network with cross-stage partial connections (C2f modules), spatial pyramid pooling (SPPF), and multi-scale feature pyramids (PANet).
- **Inference Paradigm:** Anchor-free single-stage regression directly predicting bounding box coordinates and 80-class COCO probability distributions.

---

### 6.2 Empirical Evaluation, Parameter Tuning & Failure Analysis

#### 6.2.1 Parameter Tuning Study
The Viola–Jones algorithm (`cv2.CascadeClassifier.detectMultiScale`) depends critically on three interdependent parameters:
1. **`scaleFactor`:** Controls how much the image size is reduced at each scale in the multi-scale image pyramid. A small step (e.g. 1.05 or 1.10) provides dense scale coverage but increases execution time and can capture marginal background noise. A slightly larger step (e.g. 1.15) discards weak, unstable candidate windows.
2. **`minNeighbors`:** Defines how many neighboring candidate rectangles must overlap and concur to declare a positive detection. With low thresholds (`minNeighbors = 3` or `4`), spurious isolated detections pass through the cascade. Increasing `minNeighbors = 6` enforces strict consensus, discarding isolated false positives while retaining true faces.
3. **`minSize`:** Minimum bounding window dimensions. Setting `minSize = (50, 50)` eliminates small textural artifacts (such as window panes or foliage clusters).

#### 6.2.2 Why Background Patterns Produce False Positives
Under relaxed default parameters (`scaleFactor=1.1, minNeighbors=4, minSize=(30, 30)`), the classical cascade detected a false positive on a distant building window. Because Haar cascades evaluate simple rectangular intensity differences (e.g., dark region above light region for eyes vs. cheeks, and light central strip for the nose bridge), regular architectural lattices of window frames and brickwork can accidentally produce contrast values matching a face template. Without high-level semantic context, the hand-crafted decision tree triggers a false alarm.

#### 6.2.3 Parameter Tuning Results
By tuning the parameters to `scaleFactor=1.15, minNeighbors=6, minSize=(50, 50)`:
- The architectural false positive on `original_image.jpg` was completely eliminated.
- Since no genuine human faces exist in the urban landscape scene, the system correctly reports **0 detections** and displays **"Viola-Jones: No faces detected"** without drawing false bounding boxes.
- On the street video, distant and helmeted riders also produced 0 face detections, correctly highlighting the limitation of frontal Haar cascades under occlusion and oblique angles.

#### Table 3: Object Recognition Paradigm Comparison
| Attribute | Viola–Jones (Haar Cascade - Tuned) | Ultralytics YOLOv8 (YOLOv8n) |
| :--- | :--- | :--- |
| **Model Type** | Classical AdaBoost Decision Cascade | Deep Convolutional Neural Network |
| **Target Classes** | Human Frontal Faces | 80 Common Object Classes (COCO) |
| **Original Photograph Detections** | **0 Faces Detected ("No faces detected" label)** | 0 COCO objects detected |
| **Video Frame 0 Detections** | 0 Faces detected (Misses distant/helmeted riders) | **12 Multi-Class Objects** (4 persons, 4 motorcycles, 2 cars, 1 bicycle, 1 potted plant) |
| **Execution Speed** | Extremely fast on CPU (~5 ms) | ~150 ms CPU / ~8 ms GPU |
| **Computational Footprint** | Extremely low (~900 KB XML) | 6.2 MB parameter weights |

#### Genuine Failure Mode Investigation
1. **Haar Cascade False Positives under Relaxed Tuning:** Background window grids exhibit contrast ratios resembling the eye-nose bridge intensity template, leading to false detections if `minNeighbors` is set too low ($<5$).
2. **Haar Cascade False Negatives in Video:** On the street video, multiple pedestrians and motorcyclists were present, but Haar Cascade detected zero faces. This authentic limitation arises because the subjects were distant, partially occluded, wearing helmets, or facing oblique angles—conditions under which rigid frontal Haar features fail.
3. **YOLOv8 Robustness:** Conversely, YOLOv8 accurately identified distant pedestrians, moving vehicles, and motorcycles, demonstrating the superior generalization of deep convolutional feature hierarchies.

---

## 7. Task 4: Video Motion Estimation

### 7.1 Methodology

#### 7.1.1 Frame Differencing
Computes the absolute pixel intensity difference between consecutive temporal frames:
$$\Delta I_t(x, y) = |I_t(x, y) - I_{t-1}(x, y)|$$
Followed by binary thresholding: $M_t(x, y) = 255$ if $\Delta I_t(x, y) > 25$, else $0$.

#### 7.1.2 Gaussian Mixture Model Background Subtraction (MOG2)
Models each individual pixel's history using a mixture of $K$ Gaussian distributions ($K \in [3, 5]$):
$$P(x_t) = \sum_{i=1}^K \omega_{i, t} \cdot \eta(x_t, \mu_{i, t}, \Sigma_{i, t})$$
MOG2 dynamically adapts Gaussian parameters over time to accommodate ambient lighting variations and identifies shadow pixels through chromaticity preservation.

#### 7.1.3 Pyramidal Lucas–Kanade Optical Flow
Tracks salient Shi-Tomasi corner points by assuming brightness constancy ($I(x, y, t) = I(x+dx, y+dy, t+dt)$) and small localized spatial displacement:
$$\begin{bmatrix} \sum I_x^2 & \sum I_x I_y \\ \sum I_x I_y & \sum I_y^2 \end{bmatrix} \begin{bmatrix} u \\ v \end{bmatrix} = -\begin{bmatrix} \sum I_x I_t \\ \sum I_y I_t \end{bmatrix}$$
Iterative pyramidal coarse-to-fine tracking accommodates multi-pixel displacements across consecutive frames.

---

### 7.2 Comparative Findings on Urban Video

#### Table 4: Video Motion Estimation Performance Comparison
| Motion Estimation Method | Underlying Principle | What It Identifies | Observed Strengths | Observed Weaknesses | Video Performance Metric |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Frame Differencing** | Temporal adjacent subtraction: $\|I_t - I_{t-1}\|$ | Instantaneous pixel intensity boundary transitions | Negligible memory; zero training latency; immediate response. | Hollow interior regions; ghosting artifacts; fails when objects pause. | Averaged **28,438 moving pixels/frame** during traffic surges. |
| **GMM Background Subtraction (MOG2)** | Multi-modal Gaussian distribution per pixel | Persistent foreground silhouettes separate from background | Robust solid masks; built-in shadow detection; adapts to illumination shifts. | Requires warm-up frames; sensitive to camera vibration. | Averaged **64,596 solid foreground pixels/frame**. |
| **Lucas–Kanade Optical Flow** | Gradient-based spatial velocity vector constraint | Trajectories and directional velocity vectors $(u, v)$ | Provides true direction and speed of individual moving objects. | Sparse tracking only; fails on textureless surfaces. | Tracked up to **120 concurrent feature trajectories** across vehicle paths. |

---

## 8. Summary of Saved Output Artifacts

The project generates structured, non-redundant artifacts organized into the following directories:

```
outputs/
├── task1_image_restoration_segmentation/
│   ├── restoration_metrics.csv                 # PSNR and SSIM quantitative measurements
│   ├── restoration_metrics.txt                 # Detailed text summary
│   ├── task1_01_original.png                   # Clean reference image
│   ├── task1_02_gaussian_noisy.png             # Simulated Gaussian noise (sigma=25)
│   ├── task1_03_sp_noisy.png                   # Simulated Salt-and-Pepper noise (4%)
│   ├── task1_04_gaussian_restored.png          # Gaussian-filtered image
│   ├── task1_05_median_restored.png            # Median-filtered image
│   ├── task1_06_restoration_comparison.png      # 6-panel restoration comparison figure
│   ├── task1_07_grayscale.png                  # Grayscale conversion
│   ├── task1_08_binary_threshold.png           # Otsu binary mask (T=120)
│   ├── task1_09_erosion.png                    # Morphological erosion
│   ├── task1_10_dilation.png                   # Morphological dilation
│   ├── task1_11_opening.png                    # Morphological opening
│   ├── task1_12_closing.png                    # Morphological closing
│   ├── task1_13_canny_edges.png                # Canny edge detection
│   ├── task1_14_distance_transform.png         # Normalized Euclidean distance transform
│   ├── task1_15_watershed_markers.png          # Connected component marker map
│   ├── task1_16_watershed_segmented.png        # Color image with red watershed boundaries
│   ├── task1_17_segmentation_comparison.png    # 9-panel morphological and segmentation plot
│   └── watershed_analysis.txt                  # Technical discussion on watershed limits
│
├── task2_feature_extraction/
│   ├── sift_matching_comparison.csv            # Quantitative keypoint and match comparison
│   ├── sift_matching_comparison.txt            # Experimental comparison log
│   ├── hog_and_sift_explanation.txt            # Theoretical analysis of HOG and SIFT
│   ├── task2_01_hog_visualization.png          # HOG gradient orientation image
│   ├── task2_02_hog_comparison.png             # Side-by-side HOG comparison plot
│   ├── task2_03_sift_keypoints_image1.png      # Image 1 rich SIFT keypoints
│   ├── task2_04_sift_keypoints_image2.png      # Image 2 rich SIFT keypoints
│   ├── task2_05_sift_keypoints_overview.png    # Keypoint side-by-side figure
│   ├── task2_06_sift_matching_original.png     # Baseline feature matching visualization
│   ├── task2_07_sift_matching_rotated.png      # 45-degree rotation matching visualization
│   ├── task2_08_sift_matching_resized.png      # 0.5x scale matching visualization
│   └── task2_09_sift_experiments_comparison.png# Combined 3-panel matching comparison figure
│
├── task3_object_detection/
│   ├── object_detection_comparison.csv         # Comparative detection table
│   ├── object_detection_analysis.txt           # Detailed failure analysis log
│   ├── yolo_detections_record.csv              # Full log of YOLO detected objects and confidences
│   ├── task3_01_haar_original_image.png        # Haar cascade detections on photograph
│   ├── task3_02_haar_video_frame_0.png         # Haar cascade on video frame 0
│   ├── task3_02_haar_video_frame_60.png        # Haar cascade on video frame 60
│   ├── task3_02_haar_video_frame_121.png       # Haar cascade on video frame 121
│   ├── task3_05_yolo_original_image.png        # YOLOv8 on original photograph
│   ├── task3_06_yolo_video_frame_0.png         # YOLOv8 on video frame 0
│   ├── task3_06_yolo_video_frame_60.png        # YOLOv8 on video frame 60
│   ├── task3_06_yolo_video_frame_121.png       # YOLOv8 on video frame 121
│   ├── task3_09_detection_comparison_overview.png # 4-panel Haar vs YOLO overview figure
│   └── task3_10_failure_analysis_visualized.png# Close-up visualization of Haar false positive
│
└── task4_motion_estimation/
    ├── motion_estimation_comparison.csv        # Quantitative motion estimation table
    ├── motion_estimation_analysis.txt          # Technical analysis of motion methods
    ├── task4_01_motion_estimation_overview.png # 4-panel motion estimation overview
    ├── task4_frame_36_01_original.png          # Keyframe 36 original
    ├── task4_frame_36_02_frame_diff.png        # Keyframe 36 frame difference
    ├── task4_frame_36_03_diff_thresh.png       # Keyframe 36 thresholded difference mask
    ├── task4_frame_36_04_gmm_mask.png          # Keyframe 36 GMM foreground mask
    ├── task4_frame_36_05_lk_flow.png           # Keyframe 36 Lucas-Kanade flow tracks
    ├── task4_frame_108_01_original.png         # Keyframe 108 original
    ├── task4_frame_108_02_frame_diff.png       # Keyframe 108 frame difference
    ├── task4_frame_108_03_diff_thresh.png      # Keyframe 108 thresholded difference mask
    ├── task4_frame_108_04_gmm_mask.png         # Keyframe 108 GMM foreground mask
    ├── task4_frame_108_05_lk_flow.png          # Keyframe 108 Lucas-Kanade flow tracks
    ├── task4_frame_181_01_original.png         # Keyframe 181 original
    ├── task4_frame_181_02_frame_diff.png       # Keyframe 181 frame difference
    ├── task4_frame_181_03_diff_thresh.png      # Keyframe 181 thresholded difference mask
    ├── task4_frame_181_04_gmm_mask.png         # Keyframe 181 GMM foreground mask
    ├── task4_frame_181_05_lk_flow.png          # Keyframe 181 Lucas-Kanade flow tracks
    ├── task4_gmm_foreground_mask.mp4           # Rendered video of GMM foreground masks
    └── task4_lucas_kanade_flow.mp4             # Rendered video of LK optical flow vectors
```

---

## 9. Limitations and Future Work

1. **Watershed Segmentation in Complex Scenes:** Purely intensity-driven watershed segmentation suffers from over-segmentation when applied to continuous atmospheric and foliage gradients. Future improvements could incorporate deep learning priors (e.g., SAM or Mask R-CNN) to guide initial marker placement.
2. **Dense Optical Flow:** While Lucas–Kanade provides fast, accurate sparse feature tracking, it does not estimate motion for uniform, textureless regions. Dense optical flow algorithms (such as Farnebäck or deep RAFT networks) could provide full per-pixel velocity fields.
3. **Domain-Specific Object Detection:** Standard YOLO models trained on COCO detect common everyday objects but omit natural atmospheric phenomena (such as rainbows or cloud types) without custom annotated datasets.

---

## 10. Conclusion

This project successfully implemented, verified, and benchmarked all required Computer Vision techniques across Units 3, 4, and 5:
- **Restoration:** Validated that median filtering is mathematically optimal for impulse noise (+8.15 dB gain), while Gaussian filtering is best suited for zero-mean additive noise (+6.59 dB gain).
- **Matching:** Demonstrated the scale and rotation invariance of SIFT, maintaining 1,137 valid correspondences under a 45° rotation and 512 correspondences under a 50% scale reduction.
- **Object Recognition:** Highlighted the transition from hand-crafted, failure-prone Haar cascades to deep convolutional YOLO networks capable of multi-scale, multi-class recognition in cluttered street scenes.
- **Motion Estimation:** Compared temporal differencing, adaptive statistical background modeling, and feature-based optical flow on authentic video data.

All tasks execute reliably via `main.py`, providing a complete, beginner-friendly, and mathematically grounded Computer Vision codebase.
