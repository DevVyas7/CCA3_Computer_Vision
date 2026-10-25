"""
CCA 3 - Practical Coding Assignment: Units 3, 4, and 5
Task 4: Video Motion Estimation
File: task4_motion_estimation.py

Author: Student
Description:
    This module implements:
    Part A: Frame Differencing
        - Consecutive frame absolute subtraction
        - Binary change thresholding to isolate instantaneous motion
    Part B: GMM-Based Background Subtraction
        - Statistical background modeling via Mixture of Gaussians (MOG2)
        - Dynamic background maintenance and shadow detection
        - Exporting representative foreground masks and video demo
    Part C: Lucas–Kanade Optical Flow
        - Feature detection via Shi-Tomasi corner detector (goodFeaturesToTrack)
        - Sparse optical flow point tracking across time (calcOpticalFlowPyrLK)
        - Trajectory visualization with motion vector drawing and video demo
    Part D: Comparative Analysis
        - Quantitative metrics and qualitative evaluation across methods
        - Saving CSV comparison tables and side-by-side visual summaries
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt


def get_video_path():
    """
    Finds and returns the existing path for the input video.
    """
    candidates = [
        os.path.join("dataset", "original_video.mp4"),
    ]
    if not any(os.path.exists(p) for p in candidates):
        if os.path.exists("dataset"):
            for f in os.listdir("dataset"):
                if f.endswith(".mp4"):
                    candidates.append(os.path.join("dataset", f))

    for p in candidates:
        if os.path.exists(p):
            return p
    return None


def get_video_properties(video_path):
    """
    Inspects video file and returns its frame rate, resolution, duration, and frame count.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    fps = cap.get(cv2.CAP_PROP_FPS)
    fps = fps if fps > 0 else 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = total_frames / fps if fps > 0 else 0.0

    cap.release()
    return {
        "fps": fps,
        "width": width,
        "height": height,
        "total_frames": total_frames,
        "duration_sec": duration_sec
    }


def run_task4(output_dir="outputs/task4_motion_estimation", show_plots=False, generate_videos=True):
    """
    Executes all operations for Task 4 and saves outputs.
    """
    print("\n" + "=" * 70)
    print("STARTING TASK 4: Video Motion Estimation")
    print("=" * 70)

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # 1. Locate and inspect video
    video_path = get_video_path()
    if video_path is None:
        error_msg = "ERROR: Missing input video in 'dataset/'. Please supply 'original_video.mp4'."
        print(error_msg)
        return False, error_msg

    props = get_video_properties(video_path)
    if props is None:
        error_msg = f"ERROR: OpenCV could not open or decode video at {video_path}."
        print(error_msg)
        return False, error_msg

    print(f"[Task 4] Video file loaded: {video_path}")
    print(f"[Task 4] Resolution: {props['width']}x{props['height']} pixels")
    print(f"[Task 4] Frame Rate: {props['fps']:.2f} FPS")
    print(f"[Task 4] Total Frames: {props['total_frames']}")
    print(f"[Task 4] Duration: {props['duration_sec']:.2f} seconds")

    # Select representative frames for saving still outputs
    key_indices = [
        int(props['total_frames'] * 0.15),
        int(props['total_frames'] * 0.45),
        int(props['total_frames'] * 0.75),
    ]
    print(f"[Task 4] Representative keyframe indices: {key_indices}")

    # Set up Video Writers if requested
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    gmm_video_path = os.path.join(output_dir, "task4_gmm_foreground_mask.mp4")
    lk_video_path = os.path.join(output_dir, "task4_lucas_kanade_flow.mp4")

    gmm_writer = None
    lk_writer = None
    if generate_videos:
        gmm_writer = cv2.VideoWriter(gmm_video_path, fourcc, props['fps'], (props['width'], props['height']))
        lk_writer = cv2.VideoWriter(lk_video_path, fourcc, props['fps'], (props['width'], props['height']))

    # Initialize MOG2 Background Subtractor
    fgbg = cv2.createBackgroundSubtractorMOG2(history=300, varThreshold=25, detectShadows=True)

    # Initialize Lucas-Kanade parameters
    feature_params = dict(maxCorners=120, qualityLevel=0.25, minDistance=10, blockSize=7)
    lk_params = dict(winSize=(15, 15), maxLevel=2,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    cap = cv2.VideoCapture(video_path)
    ret, first_frame = cap.read()
    if not ret:
        print("ERROR: Failed to read first video frame.")
        return False, "Failed to read first video frame."

    prev_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
    p0 = cv2.goodFeaturesToTrack(prev_gray, mask=None, **feature_params)
    lk_mask = np.zeros_like(first_frame)
    colors = np.random.randint(0, 255, (200, 3))

    # Containers for keyframe visualizations
    saved_visuals = {idx: {} for idx in key_indices}

    frame_idx = 1
    diff_motion_counts = []
    gmm_fg_counts = []
    lk_active_points = []

    print("\n--- Processing Video Frames for Differencing, GMM, and LK Optical Flow ---")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # -------------------------------------------------------------
        # 1. FRAME DIFFERENCING
        # -------------------------------------------------------------
        frame_diff = cv2.absdiff(prev_gray, curr_gray)
        # Binary thresholding at 25 intensity difference
        _, diff_thresh = cv2.threshold(frame_diff, 25, 255, cv2.THRESH_BINARY)
        diff_active = int(np.count_nonzero(diff_thresh))
        diff_motion_counts.append(diff_active)

        # -------------------------------------------------------------
        # 2. GMM-BASED BACKGROUND SUBTRACTION (MOG2)
        # -------------------------------------------------------------
        fg_mask = fgbg.apply(frame)
        # In MOG2 with shadows: 255 = foreground, 127 = shadow, 0 = background
        gmm_fg_active = int(np.count_nonzero(fg_mask == 255))
        gmm_fg_counts.append(gmm_fg_active)

        if gmm_writer is not None:
            fg_color = cv2.cvtColor(fg_mask, cv2.COLOR_GRAY2BGR)
            gmm_writer.write(fg_color)

        # -------------------------------------------------------------
        # 3. LUCAS-KANADE OPTICAL FLOW
        # -------------------------------------------------------------
        lk_frame = frame.copy()
        if p0 is not None and len(p0) > 0:
            p1, st, err = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, p0, None, **lk_params)

            if p1 is not None and st is not None:
                good_new = p1[st == 1]
                good_old = p0[st == 1]

                for i, (new, old) in enumerate(zip(good_new, good_old)):
                    a, b = new.ravel()
                    c, d = old.ravel()
                    color = colors[i % len(colors)].tolist()
                    lk_mask = cv2.line(lk_mask, (int(a), int(b)), (int(c), int(d)), color, 2)
                    lk_frame = cv2.circle(lk_frame, (int(a), int(b)), 4, color, -1)

                lk_composite = cv2.add(lk_frame, lk_mask)
                p0 = good_new.reshape(-1, 1, 2)
                lk_active_points.append(len(p0))
            else:
                lk_composite = lk_frame
                lk_active_points.append(0)
        else:
            lk_composite = lk_frame
            lk_active_points.append(0)

        # Re-detect features if tracked count drops below 20 or every 60 frames
        if (p0 is None or len(p0) < 20 or frame_idx % 60 == 0):
            new_corners = cv2.goodFeaturesToTrack(curr_gray, mask=None, **feature_params)
            if new_corners is not None:
                p0 = new_corners
            lk_mask = np.zeros_like(frame)  # Reset motion trails for clean visual

        if lk_writer is not None:
            lk_writer.write(lk_composite)

        # Store keyframes for still exports
        if frame_idx in key_indices:
            saved_visuals[frame_idx] = {
                "original": frame.copy(),
                "diff_gray": frame_diff.copy(),
                "diff_thresh": diff_thresh.copy(),
                "gmm_mask": fg_mask.copy(),
                "lk_vis": lk_composite.copy()
            }
            # Save individual representative files
            cv2.imwrite(os.path.join(output_dir, f"task4_frame_{frame_idx}_01_original.png"), frame)
            cv2.imwrite(os.path.join(output_dir, f"task4_frame_{frame_idx}_02_frame_diff.png"), frame_diff)
            cv2.imwrite(os.path.join(output_dir, f"task4_frame_{frame_idx}_03_diff_thresh.png"), diff_thresh)
            cv2.imwrite(os.path.join(output_dir, f"task4_frame_{frame_idx}_04_gmm_mask.png"), fg_mask)
            cv2.imwrite(os.path.join(output_dir, f"task4_frame_{frame_idx}_05_lk_flow.png"), lk_composite)

        prev_gray = curr_gray
        frame_idx += 1

    cap.release()
    if gmm_writer is not None:
        gmm_writer.release()
        print(f"  -> Generated GMM foreground mask video: {gmm_video_path}")
    if lk_writer is not None:
        lk_writer.release()
        print(f"  -> Generated Lucas-Kanade optical flow video: {lk_video_path}")

    # -------------------------------------------------------------
    # PART D: MULTI-PANEL OVERVIEW VISUALIZATION
    # -------------------------------------------------------------
    print("\n--- Part D: Comparative Overview & Analysis ---")
    rep_idx = key_indices[1]  # Middle keyframe
    rep_data = saved_visuals.get(rep_idx, {})

    if rep_data:
        fig, axes = plt.subplots(2, 2, figsize=(14, 16))

        axes[0, 0].imshow(cv2.cvtColor(rep_data["original"], cv2.COLOR_BGR2RGB))
        axes[0, 0].set_title(f"A. Video Frame {rep_idx} (Original)", fontsize=12, fontweight='bold')
        axes[0, 0].axis('off')

        axes[0, 1].imshow(rep_data["diff_thresh"], cmap='gray')
        axes[0, 1].set_title(f"B. Frame Differencing Mask (Threshold=25)\nCaptures instantaneous boundary motion", fontsize=11, fontweight='bold')
        axes[0, 1].axis('off')

        axes[1, 0].imshow(rep_data["gmm_mask"], cmap='gray')
        axes[1, 0].set_title(f"C. GMM (MOG2) Background Subtraction\nFull foreground silhouettes with shadow detection", fontsize=11, fontweight='bold')
        axes[1, 0].axis('off')

        axes[1, 1].imshow(cv2.cvtColor(rep_data["lk_vis"], cv2.COLOR_BGR2RGB))
        axes[1, 1].set_title(f"D. Lucas–Kanade Sparse Optical Flow\nVelocity vectors & motion trajectories", fontsize=11, fontweight='bold')
        axes[1, 1].axis('off')

        plt.suptitle("Task 4: Video Motion Estimation Techniques Comparison", fontsize=15, fontweight='bold')
        plt.tight_layout()
        overview_path = os.path.join(output_dir, "task4_01_motion_estimation_overview.png")
        plt.savefig(overview_path, dpi=200, bbox_inches='tight')
        if show_plots:
            plt.show()
        plt.close()
        print(f"  -> Saved motion estimation overview to: {overview_path}")

    # -------------------------------------------------------------
    # QUANTITATIVE COMPARISON TABLE
    # -------------------------------------------------------------
    comparison_records = [
        {
            "Technique": "Frame Differencing",
            "Identifies": "Instantaneous pixel intensity changes between adjacent frames (t and t-1)",
            "Observed_Strengths": "Extremely fast; zero memory requirement; immediately detects sudden fast movements.",
            "Observed_Weaknesses": "Produces ghosting/aperture artifacts; creates hollow boundaries for uniformly colored objects; fails when objects pause.",
            "Performance_On_Video": f"Detected an average of {int(np.mean(diff_motion_counts))} changing pixels per frame during vehicle and pedestrian motion."
        },
        {
            "Technique": "GMM Background Subtraction (MOG2)",
            "Identifies": "Persistent foreground objects distinct from a statistical Gaussian mixture background model",
            "Observed_Strengths": "Segments complete solid object silhouettes; robust to gradual illumination changes; detects shadows separately.",
            "Observed_Weaknesses": "Requires warm-up training period; susceptible to background camera jitter / panning.",
            "Performance_On_Video": f"Extracted solid vehicle and pedestrian shapes with an average of {int(np.mean(gmm_fg_counts))} foreground pixels."
        },
        {
            "Technique": "Lucas–Kanade Optical Flow",
            "Identifies": "2D velocity vectors (dx/dt, dy/dt) and motion trajectories of salient Shi-Tomasi feature points",
            "Observed_Strengths": "Provides exact directional velocity and trajectory information for individual vehicles/pedestrians.",
            "Observed_Weaknesses": "Sparse tracking only; assumes brightness constancy and small displacements; fails on textureless surfaces.",
            "Performance_On_Video": f"Tracked up to {int(np.max(lk_active_points))} concurrent feature points across traffic trajectories."
        }
    ]

    # Save to CSV
    csv_file = os.path.join(output_dir, "motion_estimation_comparison.csv")
    with open(csv_file, "w", encoding="utf-8") as f:
        f.write("Technique,Identifies,Observed_Strengths,Observed_Weaknesses,Performance_On_Video\n")
        for rec in comparison_records:
            f.write(f"{rec['Technique']},{rec['Identifies'].replace(',', ';')},{rec['Observed_Strengths'].replace(',', ';')},{rec['Observed_Weaknesses'].replace(',', ';')},{rec['Performance_On_Video'].replace(',', ';')}\n")

    # Save to TXT
    txt_file = os.path.join(output_dir, "motion_estimation_analysis.txt")
    with open(txt_file, "w", encoding="utf-8") as f:
        f.write("TASK 4: VIDEO MOTION ESTIMATION TECHNICAL ANALYSIS\n")
        f.write("=" * 65 + "\n\n")
        f.write(f"Video File:      {video_path}\n")
        f.write(f"Resolution:      {props['width']}x{props['height']}\n")
        f.write(f"Frame Count:     {props['total_frames']} frames @ {props['fps']:.2f} FPS\n\n")
        for rec in comparison_records:
            f.write(f"Technique:       {rec['Technique']}\n")
            f.write(f"Identifies:      {rec['Identifies']}\n")
            f.write(f"Strengths:       {rec['Observed_Strengths']}\n")
            f.write(f"Weaknesses:      {rec['Observed_Weaknesses']}\n")
            f.write(f"Observations:    {rec['Performance_On_Video']}\n")
            f.write("-" * 65 + "\n")

    print("\nMotion Estimation Summary Comparison:")
    print("-" * 80)
    for rec in comparison_records:
        print(f"[{rec['Technique']}]")
        print(f"  - What it identifies: {rec['Identifies']}")
        print(f"  - Actual performance: {rec['Performance_On_Video']}")
    print("-" * 80)

    print("\n[Task 4] All operations successfully completed!")
    print(f"[Task 4] Output files saved in directory: '{output_dir}'")
    print("=" * 70 + "\n")
    return True, "Task 4 completed successfully."


if __name__ == "__main__":
    success, msg = run_task4(show_plots=False)
    if not success:
        print(f"Execution Failed: {msg}")
