"""
CCA 3 – Practical Coding Assignment (Units 3, 4 and 5)
Main Program: main.py

Author: Student
Description:
    Menu-driven driver program allowing execution of individual tasks
    or all tasks consecutively. Validates environment, dependencies,
    and required dataset files before running any operation.
"""

import sys
import os
import argparse

# Import individual task modules
from task1_image_restoration import run_task1, get_image_path as get_img1_path
from task2_feature_extraction import run_task2, get_image_paths as get_img_pair_paths
from task3_object_detection import run_task3, get_media_paths as get_task3_paths
from task4_motion_estimation import run_task4, get_video_path


def check_task1_prerequisites():
    """Validates that Task 1 input file exists."""
    p = get_img1_path()
    if p is None:
        print("\n[PRE-CHECK ERROR] Missing primary photograph for Task 1!")
        print("Please ensure 'original_image.jpg' or 'original_image.jpeg' is in 'dataset/'.\n")
        return False
    return True


def check_task2_prerequisites():
    """Validates that Task 2 input files exist."""
    p1, p2 = get_img_pair_paths()
    if p1 is None or p2 is None:
        print("\n[PRE-CHECK ERROR] Missing photograph pair for Task 2!")
        print("Please ensure both 'original_image.jpg' and 'second_image.jpg' (or .jpeg) exist in 'dataset/'.\n")
        return False
    return True


def check_task3_prerequisites():
    """Validates that Task 3 input files exist."""
    img_p, vid_p = get_task3_paths()
    missing = []
    if img_p is None:
        missing.append("original photograph ('original_image.jpg' or .jpeg)")
    if vid_p is None:
        missing.append("video ('original_video.mp4')")

    if missing:
        print(f"\n[PRE-CHECK ERROR] Missing input file(s) for Task 3: {', '.join(missing)}!")
        print("Please place the required files in 'dataset/'.\n")
        return False
    return True


def check_task4_prerequisites():
    """Validates that Task 4 input file exists."""
    v = get_video_path()
    if v is None:
        print("\n[PRE-CHECK ERROR] Missing video for Task 4!")
        print("Please ensure 'original_video.mp4' (or an .mp4 file) exists in 'dataset/'.\n")
        return False
    return True


def print_banner():
    banner = """
======================================================================
     COMPUTER VISION ASSIGNMENT: CCA 3 (Units 3, 4, and 5)
======================================================================
  Course: Computer Vision
  Topics Covered:
    1. Image Restoration and Segmentation (Filters, Morphological, Watershed)
    2. Feature Extraction & Matching (HOG, SIFT, Rotation/Scale Invariance)
    3. Object Recognition (Viola-Jones Haar Cascade vs Ultralytics YOLOv8)
    4. Video Motion Estimation (Frame Differencing, GMM MOG2, Lucas-Kanade)
======================================================================
"""
    print(banner)


def print_menu():
    print("---------------------------- MAIN MENU ----------------------------")
    print("  1. Run Task 1: Image Restoration and Segmentation")
    print("  2. Run Task 2: Feature Extraction and Image Matching")
    print("  3. Run Task 3: Object Recognition (Viola-Jones & YOLO)")
    print("  4. Run Task 4: Video Motion Estimation")
    print("  5. Run All Tasks Consecutively (Tasks 1 to 4)")
    print("  6. Exit")
    print("-------------------------------------------------------------------")


def execute_task1():
    if not check_task1_prerequisites():
        return False
    success, msg = run_task1()
    return success


def execute_task2():
    if not check_task2_prerequisites():
        return False
    success, msg = run_task2()
    return success


def execute_task3():
    if not check_task3_prerequisites():
        return False
    success, msg = run_task3()
    return success


def execute_task4():
    if not check_task4_prerequisites():
        return False
    success, msg = run_task4()
    return success


def execute_all_tasks():
    print("\n>>> EXECUTING ALL TASKS (1 THROUGH 4) <<<\n")
    results = {}

    print("\n--- Running Task 1 ---")
    results["Task 1"] = execute_task1()

    print("\n--- Running Task 2 ---")
    results["Task 2"] = execute_task2()

    print("\n--- Running Task 3 ---")
    results["Task 3"] = execute_task3()

    print("\n--- Running Task 4 ---")
    results["Task 4"] = execute_task4()

    print("\n" + "=" * 60)
    print("EXECUTION SUMMARY OF ALL TASKS:")
    print("=" * 60)
    for task_name, res in results.items():
        status_str = "SUCCESS [✓]" if res else "FAILED / SKIPPED [✗]"
        print(f"  {task_name:<20}: {status_str}")
    print("=" * 60 + "\n")
    return all(results.values())


def main():
    # Support command line argument for non-interactive execution / testing
    parser = argparse.ArgumentParser(description="CCA 3 Computer Vision Runner")
    parser.add_argument("--task", type=str, choices=["1", "2", "3", "4", "all"],
                        help="Specify task number to run (1-4) or 'all'")
    parser.add_argument("--all", action="store_true", help="Run all tasks (1 through 4)")
    args = parser.parse_args()

    if args.all or args.task == "all":
        execute_all_tasks()
        return
    elif args.task:
        if args.task == "1":
            execute_task1()
        elif args.task == "2":
            execute_task2()
        elif args.task == "3":
            execute_task3()
        elif args.task == "4":
            execute_task4()
        return

    # Interactive Loop
    print_banner()
    while True:
        print_menu()
        choice = input("Enter your choice (1-6): ").strip()

        if choice == "1":
            execute_task1()
        elif choice == "2":
            execute_task2()
        elif choice == "3":
            execute_task3()
        elif choice == "4":
            execute_task4()
        elif choice == "5":
            execute_all_tasks()
        elif choice == "6":
            print("\nExiting CCA 3 Computer Vision System. Goodbye!\n")
            break
        else:
            print("\n[INVALID INPUT] Please enter a valid number between 1 and 6.\n")


if __name__ == "__main__":
    main()
