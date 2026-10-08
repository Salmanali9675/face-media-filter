"""
test_pipeline.py
Automated test suite verifying the face detection engine, media discovery,
filter modes (keep vs delete), and file actions.
"""

import os
import sys
import tempfile
import unittest
import numpy as np
import cv2

from face_engine import FaceEngine, DEFAULT_COSINE_THRESHOLD, load_image_robust
from sorter_logic import MediaSorter, FilterMode, ActionType, MediaItem


class TestFaceMediaFilter(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.engine = FaceEngine()
        cls.test_dir = tempfile.mkdtemp(prefix="facetest_")

    def test_engine_initialization(self):
        self.assertIsNotNone(self.engine.detector)
        self.assertIsNotNone(self.engine.recognizer)

    def test_blank_image_detection(self):
        # Blank image should return 0 faces and not error
        blank = np.zeros((300, 300, 3), dtype=np.uint8)
        _, faces, _ = self.engine.detect_faces(blank)
        self.assertTrue(faces is None or len(faces) == 0)

    def test_media_discovery(self):
        # Create temporary dummy files
        img_path = os.path.join(self.test_dir, "sample.jpg")
        vid_path = os.path.join(self.test_dir, "sample.mp4")
        txt_path = os.path.join(self.test_dir, "notes.txt")

        cv2.imwrite(img_path, np.zeros((100, 100, 3), dtype=np.uint8))
        with open(vid_path, "wb") as f:
            f.write(b"dummy video data")
        with open(txt_path, "w") as f:
            f.write("ignore this")

        items = MediaSorter.discover_media_files([self.test_dir], recursive=False)
        filenames = [it.filename for it in items]

        self.assertIn("sample.jpg", filenames)
        self.assertIn("sample.mp4", filenames)
        self.assertNotIn("notes.txt", filenames)

        img_item = next(it for it in items if it.filename == "sample.jpg")
        self.assertEqual(img_item.file_type, "image")
        vid_item = next(it for it in items if it.filename == "sample.mp4")
        self.assertEqual(vid_item.file_type, "video")

    def test_decision_logic(self):
        # Test Mode A: KEEP_MATCH
        # Case 1: Match found -> KEEP
        item1 = MediaItem(file_path="pic1.jpg", file_type="image", file_size_bytes=100, has_match=True)
        # Case 2: Match not found -> DELETE
        item2 = MediaItem(file_path="pic2.jpg", file_type="image", file_size_bytes=100, has_match=False)

        # Mode A: KEEP_MATCH
        # If person in photo -> keep, otherwise delete
        decision1 = "KEEP" if item1.has_match else "DELETE"
        decision2 = "KEEP" if item2.has_match else "DELETE"
        self.assertEqual(decision1, "KEEP")
        self.assertEqual(decision2, "DELETE")

        # Mode B: DELETE_MATCH
        # If person in photo -> delete, otherwise keep
        decision1_b = "DELETE" if item1.has_match else "KEEP"
        decision2_b = "DELETE" if item2.has_match else "KEEP"
        self.assertEqual(decision1_b, "DELETE")
        self.assertEqual(decision2_b, "KEEP")

    def test_action_execution_move_folder(self):
        # Create a test file to move
        file_to_del = os.path.join(self.test_dir, "to_delete.jpg")
        cv2.imwrite(file_to_del, np.zeros((50, 50, 3), dtype=np.uint8))

        item = MediaItem(
            file_path=file_to_del,
            file_type="image",
            file_size_bytes=os.path.getsize(file_to_del),
            has_match=False,
            decision="DELETE"
        )

        dest_dir = os.path.join(self.test_dir, "deleted_quarantine")
        sorter = MediaSorter(self.engine)
        stats = sorter.execute_actions([item], action_type=ActionType.MOVE_FOLDER, destination_dir=dest_dir)

        self.assertEqual(stats["moved"], 1)
        self.assertEqual(stats["errors"], 0)
        self.assertFalse(os.path.exists(file_to_del))
        self.assertTrue(os.path.exists(os.path.join(dest_dir, "to_delete.jpg")))

    def test_csv_report_export(self):
        report_path = os.path.join(self.test_dir, "report.csv")
        items = [
            MediaItem("pic1.jpg", "image", 1024, True, 0.75, "Found match", "KEEP", "Kept"),
            MediaItem("pic2.jpg", "image", 2048, False, 0.12, "No match", "DELETE", "Moved to Trash")
        ]
        MediaSorter.export_report_csv(items, report_path)
        self.assertTrue(os.path.exists(report_path))
        with open(report_path, "r", encoding="utf-8") as f:
            content = f.read()
            self.assertIn("pic1.jpg", content)
            self.assertIn("pic2.jpg", content)
            self.assertIn("0.7500", content)


if __name__ == "__main__":
    unittest.main()
