"""
sorter_logic.py
Orchestrates media discovery, face matching, decision logic, and file actions (Recycle Bin / Move / Permanent Delete).
"""

import os
import shutil
import csv
from enum import Enum
from dataclasses import dataclass, field
from typing import List, Optional, Callable, Dict, Any
from pathlib import Path
import send2trash

from face_engine import FaceEngine, DEFAULT_COSINE_THRESHOLD

SUPPORTED_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif"}
SUPPORTED_VIDEO_EXTS = {".mp4", ".mkv", ".avi", ".mov", ".wmv", ".flv", ".webm", ".m4v"}


class FilterMode(str, Enum):
    """
    User-requested decision mode:
    KEEP_MATCH: If person is in the photo/video, keep it. Otherwise, delete it.
    DELETE_MATCH: If person is in the photo/video, delete it. Otherwise, keep it.
    """
    KEEP_MATCH = "keep_match"       # Whitelist: keep matched, delete unmatched
    DELETE_MATCH = "delete_match"   # Blacklist: delete matched, keep unmatched


class ActionType(str, Enum):
    """How deletion or separation is executed."""
    TRASH = "trash"                   # Move to Windows Recycle Bin (Safe, recoverable)
    MOVE_FOLDER = "move_folder"       # Move to a designated folder (No data lost)
    PERMANENT_DELETE = "permanent"   # Permanently delete from disk
    SIMULATE = "simulate"             # Dry-run / preview only


@dataclass
class MediaItem:
    file_path: str
    file_type: str                   # 'image' or 'video'
    file_size_bytes: int
    has_match: bool = False
    similarity_score: float = 0.0
    details: str = ""
    decision: str = "PENDING"        # 'KEEP' or 'DELETE'
    action_status: str = "Pending"   # 'Pending', 'Kept', 'Moved to Trash', 'Deleted', 'Error', etc.
    override_decision: Optional[str] = None  # User can override KEEP/DELETE before executing

    @property
    def filename(self) -> str:
        return os.path.basename(self.file_path)

    @property
    def formatted_size(self) -> str:
        size = self.file_size_bytes
        for unit in ["B", "KB", "MB", "GB"]:
            if size < 1024.0:
                return f"{size:.1f} {unit}"
            size /= 1024.0
        return f"{size:.1f} TB"

    @property
    def effective_decision(self) -> str:
        return self.override_decision if self.override_decision is not None else self.decision


class MediaSorter:
    """
    Discovers media files, scans them against target profile,
    and applies file actions according to selected filter mode.
    """

    def __init__(self, face_engine: Optional[FaceEngine] = None):
        self.engine = face_engine or FaceEngine()
        self.target_profile: Optional[Dict[str, Any]] = None
        self.scanned_items: List[MediaItem] = []
        self._is_cancelled = False

    def load_target_picture(self, target_image_path: str) -> Dict[str, Any]:
        """Loads and prepares the reference face embedding."""
        profile = self.engine.extract_target_profile(target_image_path)
        if profile.get("error") is None:
            self.target_profile = profile
        return profile

    @staticmethod
    def discover_media_files(sources: List[str], recursive: bool = True) -> List[MediaItem]:
        """
        Scans folders or a list of files and returns all discovered media items.
        """
        discovered: List[MediaItem] = []
        seen_paths = set()

        def _add_file(path_str: str):
            p = Path(path_str).resolve()
            if str(p) in seen_paths:
                return
            if not p.is_file():
                return

            ext = p.suffix.lower()
            if ext in SUPPORTED_IMAGE_EXTS:
                media_type = "image"
            elif ext in SUPPORTED_VIDEO_EXTS:
                media_type = "video"
            else:
                return

            try:
                size = p.stat().st_size
            except Exception:
                size = 0

            seen_paths.add(str(p))
            discovered.append(MediaItem(
                file_path=str(p),
                file_type=media_type,
                file_size_bytes=size
            ))

        for src in sources:
            src_path = Path(src).resolve()
            if src_path.is_file():
                _add_file(str(src_path))
            elif src_path.is_dir():
                if recursive:
                    for root, _, files in os.walk(str(src_path)):
                        for f in files:
                            _add_file(os.path.join(root, f))
                else:
                    for item in src_path.iterdir():
                        if item.is_file():
                            _add_file(str(item))

        return discovered

    def cancel_scan(self):
        """Signals the scanner to stop processing."""
        self._is_cancelled = True

    def scan_media_items(self,
                         items: List[MediaItem],
                         filter_mode: FilterMode,
                         threshold: float = DEFAULT_COSINE_THRESHOLD,
                         video_sample_sec: float = 1.0,
                         progress_callback: Optional[Callable[[int, int, MediaItem, str], None]] = None) -> List[MediaItem]:
        """
        Scans a list of media items against the loaded target profile.
        Determines whether each file should be KEPT or DELETED according to filter_mode.
        """
        if not self.target_profile or "embedding" not in self.target_profile:
            raise ValueError("Target picture has not been loaded. Please select a valid reference image.")

        self._is_cancelled = False
        target_embedding = self.target_profile["embedding"]
        total = len(items)
        self.scanned_items = items

        for i, item in enumerate(items):
            if self._is_cancelled:
                break

            status_msg = f"Scanning ({i+1}/{total}): {item.filename}"
            if progress_callback:
                progress_callback(i, total, item, status_msg)

            try:
                if item.file_type == "image":
                    has_match, score, details = self.engine.scan_image(
                        item.file_path,
                        target_embedding=target_embedding,
                        threshold=threshold
                    )
                else:
                    has_match, score, details = self.engine.scan_video(
                        item.file_path,
                        target_embedding=target_embedding,
                        threshold=threshold,
                        sample_interval_sec=video_sample_sec,
                        stop_on_first_match=True
                    )

                item.has_match = has_match
                item.similarity_score = score
                item.details = details

                # Apply logic based on mode:
                # Mode A: KEEP_MATCH -> If person found: KEEP, otherwise: DELETE
                # Mode B: DELETE_MATCH -> If person found: DELETE, otherwise: KEEP
                if filter_mode == FilterMode.KEEP_MATCH:
                    item.decision = "KEEP" if has_match else "DELETE"
                else:
                    item.decision = "DELETE" if has_match else "KEEP"

            except Exception as e:
                item.details = f"Scan error: {str(e)}"
                item.decision = "KEEP"  # Default to KEEP on scan error to avoid accidental loss

            if progress_callback:
                progress_callback(i + 1, total, item, f"Finished {item.filename}")

        return items

    def execute_actions(self,
                        items: List[MediaItem],
                        action_type: ActionType,
                        destination_dir: Optional[str] = None,
                        progress_callback: Optional[Callable[[int, int, MediaItem, str], None]] = None) -> Dict[str, Any]:
        """
        Executes the file operations (Recycle Bin / Move / Delete) based on the effective decisions.
        """
        total = len(items)
        stats = {
            "total": total,
            "kept": 0,
            "deleted": 0,
            "trashed": 0,
            "moved": 0,
            "errors": 0,
            "error_details": []
        }

        if action_type == ActionType.MOVE_FOLDER and destination_dir:
            os.makedirs(destination_dir, exist_ok=True)

        for i, item in enumerate(items):
            decision = item.effective_decision
            msg = f"Processing ({i+1}/{total}): {item.filename} -> {decision}"
            if progress_callback:
                progress_callback(i, total, item, msg)

            if decision == "KEEP":
                item.action_status = "Kept (No action taken)"
                stats["kept"] += 1
                continue

            # It's marked for DELETION
            if action_type == ActionType.SIMULATE:
                item.action_status = "[Simulated] Would be deleted"
                stats["deleted"] += 1

            elif action_type == ActionType.TRASH:
                try:
                    if os.path.exists(item.file_path):
                        send2trash.send2trash(item.file_path)
                        item.action_status = "Moved to Recycle Bin"
                        stats["trashed"] += 1
                        stats["deleted"] += 1
                    else:
                        item.action_status = "File not found"
                except Exception as e:
                    stats["errors"] += 1
                    err_msg = f"Failed to recycle {item.filename}: {e}"
                    stats["error_details"].append(err_msg)
                    item.action_status = f"Error: {e}"

            elif action_type == ActionType.MOVE_FOLDER:
                try:
                    if os.path.exists(item.file_path) and destination_dir:
                        dest_path = os.path.join(destination_dir, item.filename)
                        # Handle potential filename collision
                        if os.path.exists(dest_path):
                            base, ext = os.path.splitext(item.filename)
                            counter = 1
                            while os.path.exists(os.path.join(destination_dir, f"{base}_{counter}{ext}")):
                                counter += 1
                            dest_path = os.path.join(destination_dir, f"{base}_{counter}{ext}")

                        shutil.move(item.file_path, dest_path)
                        item.action_status = f"Moved to {os.path.basename(destination_dir)}"
                        stats["moved"] += 1
                        stats["deleted"] += 1
                    else:
                        item.action_status = "File not found"
                except Exception as e:
                    stats["errors"] += 1
                    err_msg = f"Failed to move {item.filename}: {e}"
                    stats["error_details"].append(err_msg)
                    item.action_status = f"Error: {e}"

            elif action_type == ActionType.PERMANENT_DELETE:
                try:
                    if os.path.exists(item.file_path):
                        os.remove(item.file_path)
                        item.action_status = "Permanently Deleted"
                        stats["deleted"] += 1
                    else:
                        item.action_status = "File not found"
                except Exception as e:
                    stats["errors"] += 1
                    err_msg = f"Failed to delete {item.filename}: {e}"
                    stats["error_details"].append(err_msg)
                    item.action_status = f"Error: {e}"

            if progress_callback:
                progress_callback(i + 1, total, item, f"Completed {item.filename}")

        return stats

    @staticmethod
    def export_report_csv(items: List[MediaItem], output_csv_path: str):
        """Exports detailed scanning and action logs to a CSV file."""
        with open(output_csv_path, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "Filename",
                "Full Path",
                "Type",
                "Size",
                "Target Present",
                "Similarity Score",
                "Decision",
                "Action Status",
                "Details"
            ])
            for it in items:
                writer.writerow([
                    it.filename,
                    it.file_path,
                    it.file_type,
                    it.formatted_size,
                    "YES" if it.has_match else "NO",
                    f"{it.similarity_score:.4f}",
                    it.effective_decision,
                    it.action_status,
                    it.details
                ])
