"""
CCA 3 - Practical Coding Assignment: Units 3, 4, and 5
Task 1: Image Restoration and Segmentation
File: task1_image_restoration.py

Author: Student
Description:
    This module implements:
    Part A: Image Restoration
        1. Loading the original photograph
        2. Adding Gaussian and Salt-and-Pepper noise
        3. Applying Gaussian filtering (for Gaussian noise) and Median filtering (for S&P noise)
        4. Computing Quantitative metrics: PSNR (Peak Signal-to-Noise Ratio) and SSIM (Structural Similarity Index)
        5. Generating comparison visualizations
    Part B: Morphological Processing & Segmentation
        1. Grayscale conversion
        2. Binary thresholding (Otsu's method)
        3. Morphological operations: Erosion, Dilation, Opening, Closing
        4. Canny edge detection
        5. Marker-based Watershed segmentation
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as psnr_metric
from skimage.metrics import structural_similarity as ssim_metric


def get_image_path(default_paths=None):
    """
    Finds and returns the existing path for the primary original image.
    Supports both .jpg and .jpeg extensions gracefully.
    """
    if default_paths is None:
        default_paths = [
            os.path.join("dataset", "original_image.jpg"),
            os.path.join("dataset", "original_image.jpeg"),
        ]

    for p in default_paths:
        if os.path.exists(p):
            return p
    return None


def add_gaussian_noise(image, mean=0.0, sigma=25.0):
    """
    Adds zero-mean Gaussian noise to an image.
    Gaussian noise models thermal and electronic sensor noise.
    """
    noisy = image.astype(np.float32) + np.random.normal(mean, sigma, image.shape)
    noisy = np.clip(noisy, 0, 255).astype(np.uint8)
    return noisy


def add_salt_and_pepper_noise(image, salt_prob=0.02, pepper_prob=0.02):
    """
    Adds impulse (Salt-and-Pepper) noise to an image.
    Salt (white pixels) and Pepper (black pixels) model transmission errors.
    """
    noisy = image.copy()
    total_pixels = image.shape[0] * image.shape[1]

    # Salt noise (white pixels = 255)
    num_salt = int(total_pixels * salt_prob)
    salt_y = np.random.randint(0, image.shape[0], num_salt)
    salt_x = np.random.randint(0, image.shape[1], num_salt)
    noisy[salt_y, salt_x] = 255

    # Pepper noise (black pixels = 0)
    num_pepper = int(total_pixels * pepper_prob)
    pepper_y = np.random.randint(0, image.shape[0], num_pepper)
    pepper_x = np.random.randint(0, image.shape[1], num_pepper)
    noisy[pepper_y, pepper_x] = 0

    return noisy


def compute_metrics(original, processed):
    """
    Computes PSNR and SSIM between the original clean image and a processed image.
    Uses multi-channel settings for color images.
    """
    score_psnr = psnr_metric(original, processed, data_range=255)
    score_ssim = ssim_metric(original, processed, channel_axis=2, data_range=255)
    return float(score_psnr), float(score_ssim)


def perform_watershed_segmentation(bgr_image):
    """
    Performs marker-controlled watershed segmentation on the input BGR image.
    Returns:
        segmented_img: Color image with watershed boundaries drawn in red
        markers_vis: Visualizable marker labels
        dist_transform_vis: Visualizable distance transform
    """
    gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
    
    # 1. Denoise slightly and compute Otsu threshold
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # 2. Noise removal via opening
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

    # 3. Sure background area via dilation
    sure_bg = cv2.dilate(opening, kernel, iterations=3)

    # 4. Sure foreground area via Euclidean distance transform
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    dist_max = dist_transform.max()
    if dist_max > 0:
        _, sure_fg = cv2.threshold(dist_transform, 0.4 * dist_max, 255, 0)
    else:
        sure_fg = np.zeros_like(opening)
    sure_fg = np.uint8(sure_fg)

    # 5. Finding unknown region
    unknown = cv2.subtract(sure_bg, sure_fg)

    # 6. Marker labeling using connected components
    num_features, markers = cv2.connectedComponents(sure_fg)

    # Add 1 to all labels so that background is not 0, but 1
    markers = markers + 1

    # Mark the unknown region with 0
    markers[unknown == 255] = 0

    # 7. Apply watershed
    watershed_input = bgr_image.copy()
    markers_result = cv2.watershed(watershed_input, markers)

    # Mark boundary pixels (where markers == -1) in bright red
    segmented_img = bgr_image.copy()
    segmented_img[markers_result == -1] = [0, 0, 255]

    # Normalize distance transform for visualization
    dist_transform_vis = cv2.normalize(dist_transform, None, 0, 255, cv2.NORM_MINMAX)
    dist_transform_vis = np.uint8(dist_transform_vis)

    # Colorize markers for visualization
    markers_vis = np.uint8(cv2.normalize(markers_result, None, 0, 255, cv2.NORM_MINMAX))
    markers_vis = cv2.applyColorMap(markers_vis, cv2.COLORMAP_JET)

    return segmented_img, markers_vis, dist_transform_vis, markers_result


def run_task1(output_dir="outputs/task1_image_restoration_segmentation", show_plots=False):
    """
    Executes all operations for Task 1 and saves outputs.
    """
    print("\n" + "=" * 70)
    print("STARTING TASK 1: Image Restoration and Segmentation")
    print("=" * 70)

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # 1. Locate and load the primary photograph
    image_path = get_image_path()
    if image_path is None:
        error_msg = (
            "ERROR: Missing input photograph! Please place 'original_image.jpg' or "
            "'original_image.jpeg' into the 'dataset/' folder."
        )
        print(error_msg)
        return False, error_msg

    print(f"[Task 1] Successfully loaded input image from: {image_path}")
    original_bgr = cv2.imread(image_path)
    if original_bgr is None:
        error_msg = f"ERROR: OpenCV could not decode image at {image_path}."
        print(error_msg)
        return False, error_msg

    h, w, c = original_bgr.shape
    print(f"[Task 1] Image Dimensions: {w}x{h} pixels, {c} channels")

    # Save original copy
    cv2.imwrite(os.path.join(output_dir, "task1_01_original.png"), original_bgr)

    # -------------------------------------------------------------
    # PART A: IMAGE RESTORATION
    # -------------------------------------------------------------
    print("\n--- Part A: Image Restoration (Gaussian & Salt-and-Pepper) ---")
    np.random.seed(42)  # For reproducible noise experiments

    # 2. Add Gaussian noise
    gaussian_noisy = add_gaussian_noise(original_bgr, mean=0, sigma=25)
    cv2.imwrite(os.path.join(output_dir, "task1_02_gaussian_noisy.png"), gaussian_noisy)
    print("  -> Added Gaussian Noise (mean=0, sigma=25)")

    # 3. Add Salt-and-Pepper noise
    sp_noisy = add_salt_and_pepper_noise(original_bgr, salt_prob=0.02, pepper_prob=0.02)
    cv2.imwrite(os.path.join(output_dir, "task1_03_sp_noisy.png"), sp_noisy)
    print("  -> Added Salt-and-Pepper Noise (2% salt, 2% pepper)")

    # 4. Gaussian Filtering on Gaussian-noisy image
    # Kernel size 5x5, sigma 1.2
    gaussian_restored = cv2.GaussianBlur(gaussian_noisy, (5, 5), sigmaX=1.2)
    cv2.imwrite(os.path.join(output_dir, "task1_04_gaussian_restored.png"), gaussian_restored)
    print("  -> Applied Gaussian Filter (5x5 kernel, sigma=1.2) to Gaussian-noisy image")

    # 5. Median Filtering on Salt-and-Pepper noisy image
    # Kernel size 5
    median_restored = cv2.medianBlur(sp_noisy, 5)
    cv2.imwrite(os.path.join(output_dir, "task1_05_median_restored.png"), median_restored)
    print("  -> Applied Median Filter (kernel size 5) to Salt-and-Pepper noisy image")

    # Also compute cross-filtering to demonstrate filter suitability
    gaussian_on_sp = cv2.GaussianBlur(sp_noisy, (5, 5), sigmaX=1.2)
    median_on_gauss = cv2.medianBlur(gaussian_noisy, 5)

    # 6. Quantitative Evaluation (PSNR and SSIM)
    psnr_gauss_noisy, ssim_gauss_noisy = compute_metrics(original_bgr, gaussian_noisy)
    psnr_gauss_rest, ssim_gauss_rest = compute_metrics(original_bgr, gaussian_restored)
    psnr_gauss_med, ssim_gauss_med = compute_metrics(original_bgr, median_on_gauss)

    psnr_sp_noisy, ssim_sp_noisy = compute_metrics(original_bgr, sp_noisy)
    psnr_sp_rest, ssim_sp_rest = compute_metrics(original_bgr, median_restored)
    psnr_sp_gauss, ssim_sp_gauss = compute_metrics(original_bgr, gaussian_on_sp)

    print("\nQuantitative Restoration Metrics (Against Clean Original):")
    print(f"  * Gaussian Noisy:          PSNR = {psnr_gauss_noisy:6.2f} dB, SSIM = {ssim_gauss_noisy:.4f}")
    print(f"  * Gaussian Restored (Gaus):PSNR = {psnr_gauss_rest:6.2f} dB, SSIM = {ssim_gauss_rest:.4f}  [Primary]")
    print(f"  * Gaussian Filtered (Med): PSNR = {psnr_gauss_med:6.2f} dB, SSIM = {ssim_gauss_med:.4f}")
    print(f"  * S&P Noisy:               PSNR = {psnr_sp_noisy:6.2f} dB, SSIM = {ssim_sp_noisy:.4f}")
    print(f"  * S&P Restored (Median):   PSNR = {psnr_sp_rest:6.2f} dB, SSIM = {ssim_sp_rest:.4f}  [Primary]")
    print(f"  * S&P Filtered (Gaussian): PSNR = {psnr_sp_gauss:6.2f} dB, SSIM = {ssim_sp_gauss:.4f}")

    # Save metrics to CSV and TXT
    csv_path = os.path.join(output_dir, "restoration_metrics.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("Condition,Filter_Applied,PSNR_dB,SSIM\n")
        f.write(f"Gaussian_Noisy,None,{psnr_gauss_noisy:.2f},{ssim_gauss_noisy:.4f}\n")
        f.write(f"Gaussian_Noisy,Gaussian_Filter_5x5,{psnr_gauss_rest:.2f},{ssim_gauss_rest:.4f}\n")
        f.write(f"Gaussian_Noisy,Median_Filter_5,{psnr_gauss_med:.2f},{ssim_gauss_med:.4f}\n")
        f.write(f"Salt_and_Pepper_Noisy,None,{psnr_sp_noisy:.2f},{ssim_sp_noisy:.4f}\n")
        f.write(f"Salt_and_Pepper_Noisy,Median_Filter_5,{psnr_sp_rest:.2f},{ssim_sp_rest:.4f}\n")
        f.write(f"Salt_and_Pepper_Noisy,Gaussian_Filter_5x5,{psnr_sp_gauss:.2f},{ssim_sp_gauss:.4f}\n")

    txt_path = os.path.join(output_dir, "restoration_metrics.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("TASK 1: IMAGE RESTORATION METRICS SUMMARY\n")
        f.write("=" * 45 + "\n")
        f.write(f"Input image: {image_path}\n")
        f.write(f"Dimensions: {w}x{h} pixels\n\n")
        f.write("Metric Details:\n")
        f.write(f"1. Gaussian Noise (sigma=25):\n")
        f.write(f"   - Raw Noisy:        PSNR = {psnr_gauss_noisy:6.2f} dB | SSIM = {ssim_gauss_noisy:.4f}\n")
        f.write(f"   - Gaussian Filter:  PSNR = {psnr_gauss_rest:6.2f} dB | SSIM = {ssim_gauss_rest:.4f} (Effective)\n")
        f.write(f"   - Median Filter:    PSNR = {psnr_gauss_med:6.2f} dB | SSIM = {ssim_gauss_med:.4f}\n\n")
        f.write(f"2. Salt-and-Pepper Noise (4% total impulse):\n")
        f.write(f"   - Raw Noisy:        PSNR = {psnr_sp_noisy:6.2f} dB | SSIM = {ssim_sp_noisy:.4f}\n")
        f.write(f"   - Median Filter:    PSNR = {psnr_sp_rest:6.2f} dB | SSIM = {ssim_sp_rest:.4f} (Superior restoration)\n")
        f.write(f"   - Gaussian Filter:  PSNR = {psnr_sp_gauss:6.2f} dB | SSIM = {ssim_sp_gauss:.4f} (Smears impulses)\n")

    # 7. Comparison Visualization for Restoration
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes[0, 0].imshow(cv2.cvtColor(original_bgr, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title("1. Original Clean Photograph", fontsize=12, fontweight='bold')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(cv2.cvtColor(gaussian_noisy, cv2.COLOR_BGR2RGB))
    axes[0, 1].set_title(f"2. Gaussian Noise (σ=25)\nPSNR: {psnr_gauss_noisy:.2f}dB | SSIM: {ssim_gauss_noisy:.3f}", fontsize=11)
    axes[0, 1].axis('off')

    axes[0, 2].imshow(cv2.cvtColor(gaussian_restored, cv2.COLOR_BGR2RGB))
    axes[0, 2].set_title(f"3. Gaussian Filter Restored\nPSNR: {psnr_gauss_rest:.2f}dB | SSIM: {ssim_gauss_rest:.3f}", fontsize=11, color='darkgreen')
    axes[0, 2].axis('off')

    axes[1, 0].imshow(cv2.cvtColor(original_bgr, cv2.COLOR_BGR2RGB))
    axes[1, 0].set_title("4. Original Reference", fontsize=12, fontweight='bold')
    axes[1, 0].axis('off')

    axes[1, 1].imshow(cv2.cvtColor(sp_noisy, cv2.COLOR_BGR2RGB))
    axes[1, 1].set_title(f"5. Salt & Pepper Noise (4%)\nPSNR: {psnr_sp_noisy:.2f}dB | SSIM: {ssim_sp_noisy:.3f}", fontsize=11)
    axes[1, 1].axis('off')

    axes[1, 2].imshow(cv2.cvtColor(median_restored, cv2.COLOR_BGR2RGB))
    axes[1, 2].set_title(f"6. Median Filter Restored\nPSNR: {psnr_sp_rest:.2f}dB | SSIM: {ssim_sp_rest:.3f}", fontsize=11, color='darkgreen')
    axes[1, 2].axis('off')

    plt.suptitle("Task 1A: Image Noise Addition and Restoration Comparison", fontsize=16, fontweight='bold')
    plt.tight_layout()
    comparison_path = os.path.join(output_dir, "task1_06_restoration_comparison.png")
    plt.savefig(comparison_path, dpi=200, bbox_inches='tight')
    if show_plots:
        plt.show()
    plt.close()
    print(f"  -> Saved restoration comparison to: {comparison_path}")

    # -------------------------------------------------------------
    # PART B: MORPHOLOGICAL PROCESSING AND SEGMENTATION
    # -------------------------------------------------------------
    print("\n--- Part B: Morphological Processing and Watershed Segmentation ---")

    # 1. Grayscale Conversion
    gray = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(os.path.join(output_dir, "task1_07_grayscale.png"), gray)
    print("  -> 1. Grayscale conversion completed")

    # 2. Binary Thresholding (Otsu's method)
    # Applying Otsu thresholding automatically calculates the optimal threshold value
    otsu_val, binary_mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cv2.imwrite(os.path.join(output_dir, "task1_08_binary_threshold.png"), binary_mask)
    print(f"  -> 2. Binary Otsu thresholding calculated (threshold={otsu_val:.1f})")

    # Structuring element for morphological operations
    morph_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

    # 3. Erosion
    eroded = cv2.erode(binary_mask, morph_kernel, iterations=1)
    cv2.imwrite(os.path.join(output_dir, "task1_09_erosion.png"), eroded)
    print("  -> 3. Erosion performed (5x5 rectangular kernel)")

    # 4. Dilation
    dilated = cv2.dilate(binary_mask, morph_kernel, iterations=1)
    cv2.imwrite(os.path.join(output_dir, "task1_10_dilation.png"), dilated)
    print("  -> 4. Dilation performed (5x5 rectangular kernel)")

    # 5. Morphological Opening (Erosion followed by Dilation: removes small bright noise)
    opened = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, morph_kernel)
    cv2.imwrite(os.path.join(output_dir, "task1_11_opening.png"), opened)
    print("  -> 5. Morphological Opening performed")

    # 6. Morphological Closing (Dilation followed by Erosion: closes small dark holes)
    closed = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, morph_kernel)
    cv2.imwrite(os.path.join(output_dir, "task1_12_closing.png"), closed)
    print("  -> 6. Morphological Closing performed")

    # 7. Canny Edge Detection
    # Using thresholds: lower=100, upper=200
    canny_edges = cv2.Canny(gray, 100, 200)
    cv2.imwrite(os.path.join(output_dir, "task1_13_canny_edges.png"), canny_edges)
    print("  -> 7. Canny Edge Detection generated (thresholds: 100, 200)")

    # 8. Watershed Segmentation
    watershed_segmented, markers_vis, dist_transform_vis, markers_raw = perform_watershed_segmentation(original_bgr)
    cv2.imwrite(os.path.join(output_dir, "task1_14_distance_transform.png"), dist_transform_vis)
    cv2.imwrite(os.path.join(output_dir, "task1_15_watershed_markers.png"), markers_vis)
    cv2.imwrite(os.path.join(output_dir, "task1_16_watershed_segmented.png"), watershed_segmented)
    print("  -> 8. Marker-based Watershed Segmentation executed and boundaries drawn")

    # Save Watershed limitations explanation
    watershed_notes_path = os.path.join(output_dir, "watershed_analysis.txt")
    with open(watershed_notes_path, "w", encoding="utf-8") as f:
        f.write("TASK 1B: WATERSHED SEGMENTATION PERFORMANCE & ANALYSIS\n")
        f.write("=" * 60 + "\n\n")
        f.write("Dataset Nature: Complex outdoor urban landscape with sky, clouds, rainbow, buildings, and trees.\n\n")
        f.write("Observations on Watershed Results:\n")
        f.write("1. Over-segmentation tendency:\n")
        f.write("   - The watershed algorithm treats pixel intensity as a topographic surface.\n")
        f.write("   - In this photograph, subtle cloud gradients, atmospheric rainbow dispersion, and\n")
        f.write("     tree texture produce numerous local minima.\n")
        f.write("   - Consequently, watershed creates multiple subdivided basins in the sky and foliage,\n")
        f.write("     leading to characteristic over-segmentation rather than clean single-object boundaries.\n\n")
        f.write("2. Building Horizon vs Sky separation:\n")
        f.write("   - The sharp contrast between building silhouettes and the sky is effectively identified\n")
        f.write("     as major catchment divides.\n\n")
        f.write("3. Practical Engineering Implication:\n")
        f.write("   - Watershed operates best on high-contrast, well-isolated objects (such as biomedical cells\n")
        f.write("     or coins on a flat backdrop) where Euclidean distance transform provides distinct centers.\n")
        f.write("   - For complex outdoor natural scenes, semantic deep learning segmentation (e.g. Mask R-CNN / YOLO-Seg)\n")
        f.write("     is far superior to purely intensity/gradient-driven morphological watershed.\n")

    # 9. Comparison Visualization for Morphological & Segmentation
    fig, axes = plt.subplots(3, 3, figsize=(16, 16))
    
    axes[0, 0].imshow(cv2.cvtColor(original_bgr, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title("1. Original Image", fontweight='bold')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(gray, cmap='gray')
    axes[0, 1].set_title("2. Grayscale Conversion", fontweight='bold')
    axes[0, 1].axis('off')

    axes[0, 2].imshow(binary_mask, cmap='gray')
    axes[0, 2].set_title(f"3. Otsu Binary Threshold (T={otsu_val:.1f})", fontweight='bold')
    axes[0, 2].axis('off')

    axes[1, 0].imshow(eroded, cmap='gray')
    axes[1, 0].set_title("4. Morphological Erosion", fontweight='bold')
    axes[1, 0].axis('off')

    axes[1, 1].imshow(dilated, cmap='gray')
    axes[1, 1].set_title("5. Morphological Dilation", fontweight='bold')
    axes[1, 1].axis('off')

    axes[1, 2].imshow(opened, cmap='gray')
    axes[1, 2].set_title("6. Opening (Noise Cleansing)", fontweight='bold')
    axes[1, 2].axis('off')

    axes[2, 0].imshow(closed, cmap='gray')
    axes[2, 0].set_title("7. Closing (Hole Filling)", fontweight='bold')
    axes[2, 0].axis('off')

    axes[2, 1].imshow(canny_edges, cmap='gray')
    axes[2, 1].set_title("8. Canny Edge Detection", fontweight='bold')
    axes[2, 1].axis('off')

    axes[2, 2].imshow(cv2.cvtColor(watershed_segmented, cv2.COLOR_BGR2RGB))
    axes[2, 2].set_title("9. Watershed Boundaries (Red)", fontweight='bold')
    axes[2, 2].axis('off')

    plt.suptitle("Task 1B: Morphological Processing & Segmentation Overview", fontsize=16, fontweight='bold')
    plt.tight_layout()
    seg_comparison_path = os.path.join(output_dir, "task1_17_segmentation_comparison.png")
    plt.savefig(seg_comparison_path, dpi=200, bbox_inches='tight')
    if show_plots:
        plt.show()
    plt.close()
    print(f"  -> Saved segmentation comparison to: {seg_comparison_path}")

    print("\n[Task 1] All operations successfully completed!")
    print(f"[Task 1] Output files saved in directory: '{output_dir}'")
    print("=" * 70 + "\n")
    return True, "Task 1 completed successfully."


if __name__ == "__main__":
    success, msg = run_task1(show_plots=False)
    if not success:
        print(f"Execution Failed: {msg}")
