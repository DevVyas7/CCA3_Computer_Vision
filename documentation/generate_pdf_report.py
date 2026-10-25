"""
CCA 3 – Practical Coding Assignment (Units 3, 4 and 5)
Automated PDF Technical Report Generator
File: documentation/generate_pdf_report.py

Author: Student
Description:
    Compiles the experimental findings, quantitative tables, and generated
    output figures into a publication-ready academic PDF report using ReportLab.
"""

import os
import sys
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak, KeepTogether, HRFlowable
)
from reportlab.pdfgen import canvas


class NumberedCanvas(canvas.Canvas):
    """
    Two-pass canvas to dynamically compute and display total page count.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_page_decorations(self, page_count):
        if self._pageNumber > 1:
            self.saveState()
            self.setFont("Helvetica", 9)
            self.setStrokeColor(colors.HexColor("#CBD5E1"))
            self.setLineWidth(0.5)

            # Running Header
            self.line(54, 750, 558, 750)
            self.drawString(54, 755, "CCA 3 – Practical Coding Assignment | Computer Vision")
            self.drawRightString(558, 755, "Units 3, 4 and 5 Technical Report")

            # Running Footer
            self.line(54, 45, 558, 45)
            self.drawString(54, 32, "Confidential & Academic Submission")
            self.drawRightString(558, 32, f"Page {self._pageNumber} of {page_count}")
            self.restoreState()


def build_pdf_report(output_pdf_path="documentation/CCA3_Technical_Report.pdf"):
    """
    Builds the formal academic PDF technical report.
    """
    os.makedirs(os.path.dirname(output_pdf_path), exist_ok=True)
    doc = SimpleDocTemplate(
        output_pdf_path,
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=54,
        bottomMargin=54
    )

    styles = getSampleStyleSheet()

    # Custom Typography Styles
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=24,
        leading=28,
        textColor=colors.HexColor("#1E293B"),
        alignment=1,  # Center
        spaceAfter=15
    )

    subtitle_style = ParagraphStyle(
        'DocSubtitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=13,
        leading=17,
        textColor=colors.HexColor("#3B82F6"),
        alignment=1,
        spaceAfter=25
    )

    h1_style = ParagraphStyle(
        'Heading1_Custom',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=15,
        leading=19,
        textColor=colors.HexColor("#1E3A8A"),
        spaceBefore=14,
        spaceAfter=8,
        keepWithNext=True
    )

    h2_style = ParagraphStyle(
        'Heading2_Custom',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=12,
        leading=15,
        textColor=colors.HexColor("#2563EB"),
        spaceBefore=10,
        spaceAfter=5,
        keepWithNext=True
    )

    body_style = ParagraphStyle(
        'Body_Custom',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=10,
        leading=14,
        textColor=colors.HexColor("#334155"),
        spaceAfter=8
    )

    caption_style = ParagraphStyle(
        'Caption_Custom',
        parent=styles['Normal'],
        fontName='Helvetica-Oblique',
        fontSize=9,
        leading=12,
        textColor=colors.HexColor("#64748B"),
        alignment=1,
        spaceAfter=12
    )

    table_header_style = ParagraphStyle(
        'TableHeader',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=9,
        leading=11,
        textColor=colors.white,
        alignment=1
    )

    table_cell_style = ParagraphStyle(
        'TableCell',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8.5,
        leading=11,
        textColor=colors.HexColor("#1E293B")
    )

    story = []

    # =========================================================
    # 1. TITLE PAGE
    # =========================================================
    story.append(Spacer(1, 40))
    story.append(Paragraph("COMPUTER VISION LABORATORY", subtitle_style))
    story.append(Paragraph("CCA 3 – Practical Coding Assignment", title_style))
    story.append(Paragraph("Units 3, 4, and 5: Comprehensive Technical Report", ParagraphStyle('Sub', parent=subtitle_style, fontSize=12, textColor=colors.HexColor("#475569"))))
    story.append(HRFlowable(width="80%", thickness=2, color=colors.HexColor("#3B82F6"), spaceAfter=30))

    meta_text = """
    <b>Candidate Name:</b> [Student Name Placeholder]<br/>
    <b>Enrollment / Roll ID:</b> [Student ID Placeholder]<br/>
    <b>Course Title:</b> Computer Vision (CCA 3)<br/>
    <b>Department:</b> Department of Computer Science & Engineering<br/>
    <b>Institution:</b> [College / University Placeholder]<br/>
    <b>Submission Date:</b> October 2026<br/>
    <b>GitHub Repository:</b> https://github.com/username/CCA3_Computer_Vision<br/>
    <b>Walkthrough Video:</b> https://youtube.com/watch?v=placeholder
    """
    meta_p = Paragraph(meta_text, ParagraphStyle('Meta', parent=body_style, fontSize=10, leading=16, alignment=1))
    story.append(meta_p)
    story.append(Spacer(1, 50))

    exec_summary = """
    <b>Executive Summary:</b><br/>
    This report presents the implementation and rigorous experimental evaluation of four foundational 
    computer vision domains: (1) Image Restoration and Morphological Segmentation, (2) Feature Extraction 
    and Image Matching with SIFT/HOG, (3) Object Recognition comparing Viola–Jones Haar Cascades with 
    Deep Convolutional YOLOv8, and (4) Video Motion Estimation via Frame Differencing, GMM MOG2, and 
    Lucas–Kanade Optical Flow. All experiments were conducted using authentic photographs of an urban scene 
    with a rainbow and real-world street traffic video. All quantitative metrics (PSNR, SSIM, keypoint tallies, 
    matching counts, and detection records) represent genuine empirical observations.
    """
    story.append(Paragraph(exec_summary, ParagraphStyle('ExecBox', parent=body_style, backColor=colors.HexColor("#F1F5F9"), borderPadding=12, borderWidth=1, borderColor=colors.HexColor("#CBD5E1"))))
    story.append(PageBreak())

    # =========================================================
    # 2. INTRODUCTION & DATASET SPECIFICATIONS
    # =========================================================
    story.append(Paragraph("1. Introduction and Objectives", h1_style))
    story.append(Paragraph(
        "Modern computer vision bridges low-level signal processing with high-level semantic scene understanding. "
        "The objective of this assignment is to develop an end-to-end Python implementation exploring foundational "
        "algorithms across spatial filtering, gradient representations, feature matching under geometric transformations, "
        "classical vs. deep neural object detection, and temporal video motion estimation.", body_style
    ))

    story.append(Paragraph("2. Input Dataset Specification", h1_style))
    dataset_info = [
        [Paragraph("<b>File Name</b>", table_header_style), Paragraph("<b>Type & Dimensions</b>", table_header_style), Paragraph("<b>Scene Description & Role</b>", table_header_style)],
        [
            Paragraph("<code>original_image.jpg</code>", table_cell_style),
            Paragraph("RGB Image<br/>960x1280 pixels", table_cell_style),
            Paragraph("Primary urban panorama featuring multi-story buildings, trees, sky, clouds, and a rainbow. Used for restoration, morphological segmentation, and feature extraction.", table_cell_style)
        ],
        [
            Paragraph("<code>second_image.jpg</code>", table_cell_style),
            Paragraph("RGB Image<br/>960x1280 pixels", table_cell_style),
            Paragraph("Identical viewpoint captured under distinct ambient lighting and cloud conditions. Used for natural feature matching.", table_cell_style)
        ],
        [
            Paragraph("<code>original_video.mp4</code>", table_cell_style),
            Paragraph("MP4 Video<br/>480x854, 30 FPS, 242 frames", table_cell_style),
            Paragraph("Real-world urban roadway video containing moving vehicles, motorcycles, cyclists, and pedestrians. Used for object detection and motion estimation.", table_cell_style)
        ]
    ]
    t_dataset = Table(dataset_info, colWidths=[130, 110, 264])
    t_dataset.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#1E3A8A")),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
    ]))
    story.append(t_dataset)
    story.append(Spacer(1, 15))

    # =========================================================
    # 3. TASK 1: RESTORATION AND SEGMENTATION
    # =========================================================
    story.append(Paragraph("3. Task 1: Image Restoration and Segmentation", h1_style))
    story.append(Paragraph(
        "<b>Part A: Image Restoration:</b> Zero-mean Gaussian noise (&sigma;=25) and impulse Salt-and-Pepper noise (4% total corruption) "
        "were programmatically injected into the primary photograph. Restoration was performed using a 5x5 Gaussian filter (&sigma;=1.2) "
        "and a 5x5 non-linear Median filter. Restoration fidelity was evaluated using Peak Signal-to-Noise Ratio (PSNR) and Structural "
        "Similarity Index (SSIM) computed against the original pristine photograph.", body_style
    ))

    # Table 1: Quantitative Restoration Metrics
    task1_metrics = [
        [Paragraph("<b>Corrupted Input</b>", table_header_style), Paragraph("<b>Filter Applied</b>", table_header_style), Paragraph("<b>PSNR (dB)</b>", table_header_style), Paragraph("<b>SSIM Score</b>", table_header_style), Paragraph("<b>Assessment & Observation</b>", table_header_style)],
        [Paragraph("Gaussian Noisy (&sigma;=25)", table_cell_style), Paragraph("None (Raw Noisy)", table_cell_style), Paragraph("20.49 dB", table_cell_style), Paragraph("0.2542", table_cell_style), Paragraph("Baseline degraded image", table_cell_style)],
        [Paragraph("Gaussian Noisy (&sigma;=25)", table_cell_style), Paragraph("<b>Gaussian Filter (5x5)</b>", table_cell_style), Paragraph("<b>27.08 dB</b>", table_cell_style), Paragraph("<b>0.6857</b>", table_cell_style), Paragraph("<b>Optimal: +6.59 dB gain; smooth noise attenuation</b>", table_cell_style)],
        [Paragraph("Gaussian Noisy (&sigma;=25)", table_cell_style), Paragraph("Median Filter (5x5)", table_cell_style), Paragraph("25.95 dB", table_cell_style), Paragraph("0.6617", table_cell_style), Paragraph("Sub-optimal: suppresses noise but distorts continuous gradients", table_cell_style)],
        [Paragraph("Salt & Pepper (4% Impulse)", table_cell_style), Paragraph("None (Raw Noisy)", table_cell_style), Paragraph("19.13 dB", table_cell_style), Paragraph("0.3735", table_cell_style), Paragraph("Severe impulse speckle degradation", table_cell_style)],
        [Paragraph("Salt & Pepper (4% Impulse)", table_cell_style), Paragraph("<b>Median Filter (5x5)</b>", table_cell_style), Paragraph("<b>27.28 dB</b>", table_cell_style), Paragraph("<b>0.8943</b>", table_cell_style), Paragraph("<b>Optimal: +8.15 dB gain; near-complete impulse removal</b>", table_cell_style)],
        [Paragraph("Salt & Pepper (4% Impulse)", table_cell_style), Paragraph("Gaussian Filter (5x5)", table_cell_style), Paragraph("26.56 dB", table_cell_style), Paragraph("0.6748", table_cell_style), Paragraph("Defective: smears impulse noise into blurred halos", table_cell_style)]
    ]
    t_task1 = Table(task1_metrics, colWidths=[110, 95, 65, 65, 169])
    t_task1.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#2563EB")),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor("#F8FAFC")]),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
    ]))
    story.append(t_task1)
    story.append(Spacer(1, 10))

    # Restoration Comparison Figure
    fig_restoration = "outputs/task1_image_restoration_segmentation/task1_06_restoration_comparison.png"
    if os.path.exists(fig_restoration):
        story.append(Image(fig_restoration, width=490, height=310))
        story.append(Paragraph("<b>Figure 1:</b> Quantitative comparison of simulated noise and filtering restoration results.", caption_style))

    story.append(Paragraph(
        "<b>Part B: Morphological Processing and Watershed Segmentation:</b> Otsu thresholding yielded an optimal global threshold "
        "of T=120.0. Morphological erosion and dilation using a 5x5 rectangular structuring element successfully adjusted boundary margins. "
        "Opening effectively suppressed bright noise speckles, while closing bridged interior cavities. Canny edge detection (thresholds 100, 200) "
        "isolated prominent architectural contours.", body_style
    ))
    story.append(Paragraph(
        "<i>Watershed Limitations in Complex Landscapes:</i> Applying marker-controlled watershed demonstrated the classic over-segmentation "
        "limitation. While the algorithm cleanly delineated building silhouettes against the sky, subtle atmospheric rainbow gradients and tree foliage "
        "generated numerous local gradient minima, subdividing the continuous sky into irregular basins.", body_style
    ))

    fig_segmentation = "outputs/task1_image_restoration_segmentation/task1_17_segmentation_comparison.png"
    if os.path.exists(fig_segmentation):
        story.append(Image(fig_segmentation, width=490, height=430))
        story.append(Paragraph("<b>Figure 2:</b> Morphological transforms, edge detection, and marker-based watershed segmentation.", caption_style))

    story.append(PageBreak())

    # =========================================================
    # 4. TASK 2: FEATURE EXTRACTION AND MATCHING
    # =========================================================
    story.append(Paragraph("4. Task 2: Feature Extraction and Image Matching", h1_style))
    story.append(Paragraph(
        "<b>Part A: Histogram of Oriented Gradients (HOG):</b> Extracted using 16x16 pixel cells and 9 orientation bins, generating a "
        "63,504-dimensional descriptor vector. The visualization captures dominant building rooflines, windows, and vertical facades while "
        "leaving smooth sky regions dark, demonstrating robustness against ambient lighting changes.", body_style
    ))

    fig_hog = "outputs/task2_feature_extraction/task2_02_hog_comparison.png"
    if os.path.exists(fig_hog):
        story.append(Image(fig_hog, width=490, height=220))
        story.append(Paragraph("<b>Figure 3:</b> Primary photograph alongside HOG gradient orientation visualization.", caption_style))

    story.append(Paragraph(
        "<b>Part B-E: SIFT Keypoints and Geometric Invariance Experiments:</b> SIFT detected scale-space extrema across octaves. "
        "Keypoint matching was executed between Image 1 and Image 2 using Brute-Force matching with Lowe's ratio test (threshold 0.75). "
        "Invariance was evaluated under two rigorous transformations: in-plane 45° rotation and 50% scale reduction.", body_style
    ))

    task2_table = [
        [Paragraph("<b>Experiment Name</b>", table_header_style), Paragraph("<b>Transform Applied</b>", table_header_style), Paragraph("<b>Img 1 KPs</b>", table_header_style), Paragraph("<b>Img 2 KPs</b>", table_header_style), Paragraph("<b>Good Matches</b>", table_header_style), Paragraph("<b>Invariance Evaluation</b>", table_header_style)],
        [
            Paragraph("Original Pair (Baseline)", table_cell_style),
            Paragraph("Natural Lighting Shift", table_cell_style),
            Paragraph("4,607", table_cell_style),
            Paragraph("6,361", table_cell_style),
            Paragraph("<b>1,173</b>", table_cell_style),
            Paragraph("Robust correspondence despite noticeable sunlight and exposure variation.", table_cell_style)
        ],
        [
            Paragraph("Rotation Invariance", table_cell_style),
            Paragraph("Rotated 45 degrees", table_cell_style),
            Paragraph("4,444", table_cell_style),
            Paragraph("6,361", table_cell_style),
            Paragraph("<b>1,137</b>", table_cell_style),
            Paragraph("<b>96.9% match retention</b>; canonical orientation assignment neutralizes rotation.", table_cell_style)
        ],
        [
            Paragraph("Scale Invariance", table_cell_style),
            Paragraph("Resized 0.5x Scale", table_cell_style),
            Paragraph("1,423", table_cell_style),
            Paragraph("6,361", table_cell_style),
            Paragraph("<b>512</b>", table_cell_style),
            Paragraph("Fewer points due to resolution decimation, but valid correspondence is preserved.", table_cell_style)
        ]
    ]
    t_task2 = Table(task2_table, colWidths=[110, 95, 55, 55, 65, 124])
    t_task2.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#2563EB")),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor("#F8FAFC")]),
        ('TOPPADDING', (0, 0), (-1, -1), 5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
    ]))
    story.append(t_task2)
    story.append(Spacer(1, 10))

    fig_sift = "outputs/task2_feature_extraction/task2_09_sift_experiments_comparison.png"
    if os.path.exists(fig_sift):
        story.append(Image(fig_sift, width=490, height=430))
        story.append(Paragraph("<b>Figure 4:</b> SIFT matching under natural lighting (top), 45° rotation (middle), and 0.5x downscaling (bottom).", caption_style))

    story.append(PageBreak())

    # =========================================================
    # 5. TASK 3: OBJECT RECOGNITION & FAILURE ANALYSIS
    # =========================================================
    story.append(Paragraph("5. Task 3: Object Recognition (Viola–Jones vs. YOLOv8)", h1_style))
    story.append(Paragraph(
        "A classical sliding-window detector (Viola–Jones with Haar cascade decision trees) was directly benchmarked against a modern "
        "deep convolutional neural network (Ultralytics YOLOv8n nano). Both models were evaluated on the high-resolution photograph and "
        "selected representative video frames.", body_style
    ))

    task3_table = [
        [Paragraph("<b>Attribute</b>", table_header_style), Paragraph("<b>Viola–Jones (Haar Cascade)</b>", table_header_style), Paragraph("<b>Ultralytics YOLOv8 (YOLOv8n)</b>", table_header_style)],
        [Paragraph("Underlying Architecture", table_cell_style), Paragraph("Integral images, Haar rectangular intensity features, AdaBoost cascaded classifier", table_cell_style), Paragraph("Deep CNN with C2f feature extractors, SPPF pooling, and anchor-free detection head", table_cell_style)],
        [Paragraph("Target Classes", table_cell_style), Paragraph("Single class: Frontal Human Faces", table_cell_style), Paragraph("80 common object classes (COCO dataset)", table_cell_style)],
        [Paragraph("Photograph Detections", table_cell_style), Paragraph("<b>1 Candidate Box (Architectural False Positive)</b>", table_cell_style), Paragraph("0 COCO objects detected (True Negative)", table_cell_style)],
        [Paragraph("Video Frame 0 Detections", table_cell_style), Paragraph("0 Faces detected (Missed distant/helmeted riders)", table_cell_style), Paragraph("<b>12 Multi-Class Objects</b> (4 persons, 4 motorcycles, 2 cars, 1 bicycle, 1 potted plant)", table_cell_style)],
        [Paragraph("Inference Latency", table_cell_style), Paragraph("~5 ms per frame (CPU)", table_cell_style), Paragraph("~150 ms (CPU) / ~8 ms (GPU)", table_cell_style)],
        [Paragraph("Model Size / Weights", table_cell_style), Paragraph("~900 KB XML file", table_cell_style), Paragraph("6.2 MB PyTorch weights file", table_cell_style)]
    ]
    t_task3 = Table(task3_table, colWidths=[110, 190, 204])
    t_task3.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#2563EB")),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor("#F8FAFC")]),
        ('TOPPADDING', (0, 0), (-1, -1), 5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
    ]))
    story.append(t_task3)
    story.append(Spacer(1, 10))

    story.append(Paragraph(
        "<b>Investigation of Genuine Failure Modes:</b><br/>"
        "1. <i>Haar False Positive:</i> On the landscape photograph, Viola–Jones produced a false positive detection on the repetitive window "
        "grid of a distant building. This failure occurs because horizontal and vertical building mullions create rectangular dark-light contrast ratios "
        "that identically mimic the mathematical template of human eye sockets and nose bridges.<br/>"
        "2. <i>Haar False Negatives:</i> In the street video, multiple pedestrians and motorcycle riders were visible, yet Haar detected zero faces. "
        "Because classical cascades depend on rigid frontal symmetry, distant, tilted, or helmet-occluded faces fail activation thresholds.<br/>"
        "3. <i>YOLO Generalization:</i> YOLOv8 reliably detected distant vehicles and riders under motion blur and partial occlusion, highlighting "
        "the superiority of hierarchical deep convolutional representations.", body_style
    ))

    fig_det_overview = "outputs/task3_object_detection/task3_09_detection_comparison_overview.png"
    if os.path.exists(fig_det_overview):
        story.append(Image(fig_det_overview, width=490, height=360))
        story.append(Paragraph("<b>Figure 5:</b> Detection comparison: Haar Cascade vs YOLOv8 on the photograph and video frame 0.", caption_style))

    fig_failure = "outputs/task3_object_detection/task3_10_failure_analysis_visualized.png"
    if os.path.exists(fig_failure):
        story.append(Image(fig_failure, width=490, height=210))
        story.append(Paragraph("<b>Figure 6:</b> Close-up analysis: Haar architectural false positive (left) vs YOLO traffic detections (right).", caption_style))

    story.append(PageBreak())

    # =========================================================
    # 6. TASK 4: VIDEO MOTION ESTIMATION
    # =========================================================
    story.append(Paragraph("6. Task 4: Video Motion Estimation", h1_style))
    story.append(Paragraph(
        "Three distinct motion estimation paradigms were implemented and evaluated on the 242-frame urban street video: "
        "(1) Frame Differencing, (2) GMM-based Background Subtraction (MOG2), and (3) Lucas–Kanade Sparse Optical Flow.", body_style
    ))

    task4_table = [
        [Paragraph("<b>Technique</b>", table_header_style), Paragraph("<b>What It Identifies</b>", table_header_style), Paragraph("<b>Observed Strengths</b>", table_header_style), Paragraph("<b>Observed Weaknesses</b>", table_header_style), Paragraph("<b>Empirical Performance Metric</b>", table_header_style)],
        [
            Paragraph("Frame Differencing", table_cell_style),
            Paragraph("Instantaneous pixel intensity transitions between adjacent frames (|t - (t-1)|)", table_cell_style),
            Paragraph("Zero memory overhead; immediate detection of sudden motion.", table_cell_style),
            Paragraph("Hollow boundary artifacts; ghosting; fails when objects pause.", table_cell_style),
            Paragraph("Averaged <b>28,438 moving pixels/frame</b> during traffic motion.", table_cell_style)
        ],
        [
            Paragraph("GMM Background Subtraction (MOG2)", table_cell_style),
            Paragraph("Persistent foreground silhouettes distinct from statistical Gaussian background", table_cell_style),
            Paragraph("Extracts complete solid silhouettes; built-in shadow detection; adapts to illumination.", table_cell_style),
            Paragraph("Requires warm-up training period; sensitive to camera shake.", table_cell_style),
            Paragraph("Extracted solid vehicles/pedestrians with avg <b>64,596 foreground pixels/frame</b>.", table_cell_style)
        ],
        [
            Paragraph("Lucas–Kanade Optical Flow", table_cell_style),
            Paragraph("2D velocity vectors (dx/dt, dy/dt) and trajectories of Shi-Tomasi corners", table_cell_style),
            Paragraph("Provides true directional velocity and trajectory tracking.", table_cell_style),
            Paragraph("Sparse tracking only; assumes small displacement and brightness constancy.", table_cell_style),
            Paragraph("Tracked up to <b>120 concurrent feature trajectories</b> across traffic.", table_cell_style)
        ]
    ]
    t_task4 = Table(task4_table, colWidths=[90, 105, 105, 100, 104])
    t_task4.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#2563EB")),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor("#F8FAFC")]),
        ('TOPPADDING', (0, 0), (-1, -1), 5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
    ]))
    story.append(t_task4)
    story.append(Spacer(1, 10))

    fig_motion = "outputs/task4_motion_estimation/task4_01_motion_estimation_overview.png"
    if os.path.exists(fig_motion):
        story.append(Image(fig_motion, width=490, height=440))
        story.append(Paragraph("<b>Figure 7:</b> Comparative motion estimation overview on representative video frame.", caption_style))

    # =========================================================
    # 7. LIMITATIONS AND CONCLUSION
    # =========================================================
    story.append(Paragraph("7. Technical Limitations", h1_style))
    story.append(Paragraph(
        "1. <b>Morphological Watershed:</b> Gradient-based watershed is constrained by over-segmentation on complex natural outdoor textures. "
        "Semantic neural networks (e.g. YOLO-Seg or Mask R-CNN) are required for reliable instance boundary demarcation.<br/>"
        "2. <b>Sparse vs Dense Motion:</b> Lucas–Kanade tracking is computationally efficient but omits textureless vehicle bodies. "
        "Dense flow algorithms (e.g. Farnebäck or RAFT) offer per-pixel flow fields at higher computational cost.<br/>"
        "3. <b>Haar Cascade Geometric Fragility:</b> Haar decision trees lack non-linear feature invariant learning, failing under rotation, "
        "partial occlusion, or non-frontal perspectives.", body_style
    ))

    story.append(Paragraph("8. Conclusion", h1_style))
    story.append(Paragraph(
        "All four core tasks required by the CCA 3 assignment were fully implemented, rigorously tested, and empirically evaluated. "
        "Quantitative measurements confirmed the theoretical superiority of median filtering for impulse noise (+8.15 dB) and Gaussian filtering "
        "for thermal sensor noise (+6.59 dB). SIFT demonstrated remarkable scale and rotation invariance, maintaining over 1,100 valid matches "
        "under 45° rotation. The contrast between Viola–Jones and YOLOv8 substantiated the paradigm shift from handcrafted heuristics to deep feature "
        "learning. Finally, video motion analysis demonstrated how temporal differencing, adaptive Gaussian mixture modeling, and optical flow "
        "serve complementary roles in real-time video surveillance.", body_style
    ))

    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"Successfully generated academic PDF report at: {output_pdf_path}")
    return True


if __name__ == "__main__":
    build_pdf_report()
