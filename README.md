# 👤 Face Media Filter & Sorter

An AI-powered desktop tool that filters collections of **photos and videos** using deep learning facial recognition based on a reference picture of a target person.

---

## ✨ Features

- **Both Photos & Videos Supported**:
  - Images: `.jpg`, `.jpeg`, `.png`, `.webp`, `.bmp`, `.tiff`
  - Videos: `.mp4`, `.mkv`, `.avi`, `.mov`, `.wmv`, `.webm`, `.flv`, `.m4v`
- **Two Filtering Modes**:
  1. **KEEP Files with Target Person** (Whitelist mode): If the person is in the photo or video, keep it. Otherwise, delete/isolate it.
  2. **DELETE Files with Target Person** (Blacklist mode): If the person is in the photo or video, delete/isolate it. Otherwise, keep it.
- **Safe Deletion Options**:
  - **Recycle Bin (Default & Recommended)**: Deleted files are moved to the Windows Recycle Bin using `send2trash`, so they can be restored if needed.
  - **Move to Separate Folder**: Moves files into a quarantine/isolation directory without deleting anything.
  - **Dry Run / Simulation**: Previews what would happen without altering any files on disk.
  - **Permanent Delete**: Permanently removes files with safety confirmation.
- **Modern Desktop GUI**:
  - Built with CustomTkinter (dark theme, responsive layout).
  - Visual face preview showing the detected face bounding box and thumbnail.
  - Interactive results table with thumbnails, confidence scores, and individual KEEP/DELETE overrides.
  - Live progress bar and statistics cards.
  - Export audit reports to CSV.
- **Fast & Lightweight AI**:
  - Uses OpenCV Zoo DNN deep learning models: **YuNet** (ultra-fast face detector) and **SFace** (128-D cosine embedding recognizer).
  - No heavy 2GB PyTorch or TensorFlow runtimes required.
  - Fast video scanning with configurable temporal frame sampling and instant early-exit upon match detection.

---

## 🚀 Quick Start

### 1. Launch the Desktop GUI

You can start the GUI in any of the following ways:

#### Option A: Double-Click Batch File (Easiest on Windows)
Simply double-click:
```
run_gui.bat
```

#### Option B: From Command Prompt or PowerShell
```powershell
cd C:\Users\Salma\.gemini\antigravity\scratch\face-media-filter
python main.py
```

---

## 🖥️ Using the GUI

1. **Select Reference Photo**:
   Click **"Select Reference Photo..."** and choose a clear picture of the person you want to match. The application will detect the face and display a preview thumbnail.
2. **Select Media**:
   Click **"Select Folder"** to scan a folder of photos and videos (with optional subfolder recursion) or **"Select Files"** to pick individual files.
3. **Choose Filter Mode**:
   - `KEEP if target is in photo/video` (deletes files without the person)
   - `DELETE if target is in photo/video` (keeps files without the person)
4. **Choose Action**:
   - `Send to Recycle Bin` (Safe, recoverable)
   - `Move to a Separate Folder...`
   - `Preview Only (Dry Run)`
5. **Adjust Settings (Optional)**:
   - **Match Strictness**: Default `0.40` (SFace cosine threshold). Lower values match more aggressively; higher values are stricter.
   - **Video Frame Interval**: Default `1.0s` (samples one frame per second).
6. **Scan & Review**:
   Click **"Start Scan & Identify"**. As files are processed, review the match percentage and planned action for each file. You can click any file's action button to manually toggle between **KEEP** and **DELETE**.
7. **Apply Actions**:
   Click **"Apply & Execute File Actions"** to safely execute the chosen operations with confirmation.

---

## 💻 Command-Line Interface (CLI)

The tool also features a full CLI for scripting and batch automation:

### Examples

#### Example 1: Keep only photos/videos containing the person (Move others to Recycle Bin)
```bash
python cli.py --target "ref.jpg" --media-dir "C:\MyMedia\Photos" --mode keep_match --action trash
```

#### Example 2: Delete all photos/videos containing the person
```bash
python cli.py --target "ref.jpg" --media-dir "C:\MyMedia\Photos" --mode delete_match --action trash
```

#### Example 3: Safe Dry-Run (Simulation preview without touching files)
```bash
python cli.py --target "ref.jpg" --media-dir "C:\MyMedia\Photos" --mode keep_match --action simulate
```

#### Example 4: Move files without the person to a separate folder
```bash
python cli.py --target "ref.jpg" --media-dir "C:\MyMedia\Photos" --mode keep_match --action move --dest-dir "C:\MyMedia\Quarantine"
```

#### Example 5: Save detailed CSV report
```bash
python cli.py --target "ref.jpg" --media-dir "C:\MyMedia\Photos" --mode keep_match --action trash --report "scan_results.csv"
```

### CLI Arguments Reference

| Flag | Description | Default |
|---|---|---|
| `--target, -t` | Path to the reference photo containing the target person's face (Required) | - |
| `--media-dir, -d` | Directory containing photos and videos to scan | - |
| `--files, -f` | Specific media files to scan | - |
| `--mode, -m` | `keep_match` (whitelist) or `delete_match` (blacklist) | `keep_match` |
| `--action, -a` | `trash` (Recycle Bin), `move` (folder), `permanent` (disk delete), `simulate` | `trash` |
| `--dest-dir` | Target folder when using `--action move` | - |
| `--threshold` | Cosine similarity threshold (0.25 to 0.65) | `0.40` |
| `--video-interval` | Sampling interval in seconds for video processing | `1.0` |
| `--no-recursive` | Disable recursive subdirectory scanning | False |
| `--report, -r` | Path to write CSV audit log | - |
| `--yes, -y` | Skip interactive confirmation prompt | False |

---

## 🧪 Testing & Verification

Run the automated test suites:

```bash
# Run unit tests
python -m unittest test_pipeline.py

# Run end-to-end integration test with synthetic video and photos
python test_e2e.py
```

Try the pre-configured demo media:
```bash
python setup_demo_samples.py
python cli.py --target demo_samples/reference_person.jpg --media-dir demo_samples --mode keep_match --action simulate
```

---

## 📁 Project Structure

```
face-media-filter/
├── models/                         # Pre-trained ONNX models
│   ├── face_detection_yunet.onnx   # Fast face detector (YuNet)
│   └── face_recognition_sface.onnx # 128-D face recognizer (SFace)
├── face_engine.py                  # Core face detection, embedding & video processing
├── sorter_logic.py                 # Media scanner, Keep/Delete logic & file actions
├── gui.py                          # Modern CustomTkinter desktop interface
├── cli.py                          # Terminal interface with arguments & colors
├── main.py                         # Unified launcher (GUI / CLI)
├── run_gui.bat                     # Windows batch script to launch GUI
├── test_pipeline.py                # Unit test suite
├── test_e2e.py                     # Full end-to-end verification
├── setup_demo_samples.py           # Sample media generator for testing
├── requirements.txt                # Python dependencies
└── README.md                       # Project documentation
```
