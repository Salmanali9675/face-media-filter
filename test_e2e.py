"""
test_e2e.py
End-to-end verification of Face Media Filter with real image and video files.
Downloads two sample face images from OpenCV public samples and tests detection,
cosine matching, and decision execution.
"""

import os
import sys
import shutil
import tempfile
import urllib.request
import numpy as np
import cv2

if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

from face_engine import FaceEngine
from sorter_logic import MediaSorter, FilterMode, ActionType


def run_e2e():
    test_dir = tempfile.mkdtemp(prefix="face_e2e_")
    print(f"Running E2E tests in temporary directory: {test_dir}")

    try:
        lena_url = "https://raw.githubusercontent.com/opencv/opencv/master/samples/data/lena.jpg"
        target_path = os.path.join(test_dir, "target_ref.jpg")
        same_person_path = os.path.join(test_dir, "same_person_photo.jpg")
        blank_photo_path = os.path.join(test_dir, "nature_blank.jpg")
        video_path = os.path.join(test_dir, "test_video.mp4")

        # Download reference face
        urllib.request.urlretrieve(lena_url, target_path)

        # Create a modified copy of target as "same person"
        img = cv2.imread(target_path)
        h, w = img.shape[:2]
        # Crop 90% center then resize back to original
        cropped = img[int(h*0.05):int(h*0.95), int(w*0.05):int(w*0.95)]
        cropped_resized = cv2.resize(cropped, (w, h))
        cv2.imwrite(same_person_path, cropped_resized)

        # Create a photo without any face (exact same dimensions)
        blank = np.zeros((h, w, 3), dtype=np.uint8)
        blank[:, :, 0] = 120  # Blue background
        cv2.imwrite(blank_photo_path, blank)

        # Create a video containing the face in frames 5-15
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(video_path, fourcc, 10.0, (w, h))
        for i in range(20):
            if 5 <= i <= 15:
                out.write(img)
            else:
                out.write(blank)
        out.release()

        # Initialize Sorter
        sorter = MediaSorter()
        profile = sorter.load_target_picture(target_path)
        assert profile.get("error") is None, f"Failed to load target: {profile.get('error')}"
        print(f"[OK] Target face loaded successfully (detected {profile['face_count']} face)")

        # Discover files
        discovered = sorter.discover_media_files([test_dir])
        candidates = [it for it in discovered if it.file_path != target_path]
        print(f"[OK] Discovered {len(candidates)} media candidates")

        # --- TEST 1: MODE A (KEEP MATCH: Keep files with target, delete without) ---
        print("\n--- Testing Mode A: KEEP MATCH ---")
        results_a = sorter.scan_media_items(
            items=candidates,
            filter_mode=FilterMode.KEEP_MATCH,
            threshold=0.38,
            video_sample_sec=0.5
        )

        for it in results_a:
            print(f"  {it.filename:<25} | Match: {it.has_match!s:<5} | Score: {it.similarity_score:.3f} | Decision: {it.decision}")

        same_item = next(it for it in results_a if it.filename == "same_person_photo.jpg")
        blank_item = next(it for it in results_a if it.filename == "nature_blank.jpg")
        vid_item = next(it for it in results_a if it.filename == "test_video.mp4")

        assert same_item.has_match is True, "Same person photo should match!"
        assert same_item.decision == "KEEP", "Same person photo should be KEPT in KEEP_MATCH mode!"

        assert blank_item.has_match is False, "Blank photo should not match!"
        assert blank_item.decision == "DELETE", "Blank photo should be DELETED in KEEP_MATCH mode!"

        assert vid_item.has_match is True, "Video containing face should match!"
        assert vid_item.decision == "KEEP", "Video containing face should be KEPT in KEEP_MATCH mode!"
        print("[OK] Mode A (KEEP_MATCH) logic verified successfully!")

        # --- TEST 2: MODE B (DELETE MATCH: Delete files with target, keep without) ---
        print("\n--- Testing Mode B: DELETE MATCH ---")
        candidates_b = sorter.discover_media_files([test_dir])
        candidates_b = [it for it in candidates_b if it.file_path != target_path]

        results_b = sorter.scan_media_items(
            items=candidates_b,
            filter_mode=FilterMode.DELETE_MATCH,
            threshold=0.38,
            video_sample_sec=0.5
        )

        for it in results_b:
            print(f"  {it.filename:<25} | Match: {it.has_match!s:<5} | Score: {it.similarity_score:.3f} | Decision: {it.decision}")

        same_item_b = next(it for it in results_b if it.filename == "same_person_photo.jpg")
        blank_item_b = next(it for it in results_b if it.filename == "nature_blank.jpg")
        vid_item_b = next(it for it in results_b if it.filename == "test_video.mp4")

        assert same_item_b.decision == "DELETE", "Same person photo should be DELETED in DELETE_MATCH mode!"
        assert blank_item_b.decision == "KEEP", "Blank photo should be KEPT in DELETE_MATCH mode!"
        assert vid_item_b.decision == "DELETE", "Video containing face should be DELETED in DELETE_MATCH mode!"
        print("[OK] Mode B (DELETE_MATCH) logic verified successfully!")

        # --- TEST 3: Action Execution (Move to destination folder) ---
        dest_quarantine = os.path.join(test_dir, "quarantine_deleted")
        stats = sorter.execute_actions(
            items=results_b,
            action_type=ActionType.MOVE_FOLDER,
            destination_dir=dest_quarantine
        )
        print(f"\n[OK] Action execution completed: {stats}")
        assert stats["moved"] == 2, f"Expected 2 files moved, got {stats['moved']}"
        assert os.path.exists(os.path.join(dest_quarantine, "same_person_photo.jpg"))
        assert os.path.exists(os.path.join(dest_quarantine, "test_video.mp4"))
        assert os.path.exists(blank_photo_path), "Blank photo should have remained in original location!"
        print("[OK] File movement and isolation verified successfully!")

        print("\n=======================================================")
        print("       ALL END-TO-END VERIFICATION CHECKS PASSED       ")
        print("=======================================================")

    finally:
        shutil.rmtree(test_dir, ignore_errors=True)


if __name__ == "__main__":
    run_e2e()
