"""
setup_demo_samples.py
Creates sample demo photos and video in 'demo_samples/' for immediate testing.
"""

import os
import urllib.request
import cv2
import numpy as np

DEMO_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "demo_samples")


def setup_demo():
    os.makedirs(DEMO_DIR, exist_ok=True)
    ref_path = os.path.join(DEMO_DIR, "reference_person.jpg")
    photo_with = os.path.join(DEMO_DIR, "photo_with_target.jpg")
    photo_without = os.path.join(DEMO_DIR, "photo_without_target.jpg")
    video_with = os.path.join(DEMO_DIR, "video_with_target.mp4")

    # Download public domain sample face (Lena)
    sample_url = "https://raw.githubusercontent.com/opencv/opencv/master/samples/data/lena.jpg"
    if not os.path.exists(ref_path):
        print("Downloading sample reference photo...")
        urllib.request.urlretrieve(sample_url, ref_path)

    img = cv2.imread(ref_path)
    h, w = img.shape[:2]

    # Create photo with target
    if not os.path.exists(photo_with):
        cropped = img[int(h*0.05):int(h*0.95), int(w*0.05):int(w*0.95)]
        cv2.imwrite(photo_with, cv2.resize(cropped, (w, h)))

    # Create photo without target (landscape pattern)
    if not os.path.exists(photo_without):
        blank = np.zeros((h, w, 3), dtype=np.uint8)
        # Gradient pattern
        for y in range(h):
            blank[y, :, 0] = int(255 * (y / h))  # Blue gradient
            blank[y, :, 1] = int(180 * (1 - y / h))  # Green gradient
            blank[y, :, 2] = 50
        cv2.putText(blank, "Landscape View", (int(w*0.2), int(h*0.5)), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
        cv2.imwrite(photo_without, blank)

    # Create video containing the face
    if not os.path.exists(video_with):
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(video_with, fourcc, 10.0, (w, h))
        blank = np.zeros((h, w, 3), dtype=np.uint8)
        for i in range(25):
            if 8 <= i <= 18:
                out.write(img)
            else:
                out.write(blank)
        out.release()

    print(f"Sample demo files ready in: {DEMO_DIR}")
    print(f"  - reference_person.jpg")
    print(f"  - photo_with_target.jpg")
    print(f"  - photo_without_target.jpg")
    print(f"  - video_with_target.mp4")


if __name__ == "__main__":
    setup_demo()
