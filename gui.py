"""
gui.py
Modern desktop graphical user interface for Face Media Filter using CustomTkinter.
Provides face preview, multi-mode selection, progress feedback, results review table,
and safe execution options.
"""

import os
import sys
import threading
import tkinter as tk
from tkinter import filedialog, messagebox
from typing import List, Optional, Dict, Any
from pathlib import Path
from PIL import Image, ImageTk

import customtkinter as ctk

from face_engine import FaceEngine, DEFAULT_COSINE_THRESHOLD, load_image_robust
from sorter_logic import MediaSorter, FilterMode, ActionType, MediaItem

# Configure CustomTkinter defaults
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")


class FaceMediaFilterApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Face Media Filter & Sorter")
        self.geometry("1180x820")
        self.minsize(980, 700)

        # Initialize Logic Components
        self.engine: Optional[FaceEngine] = None
        self.sorter: Optional[MediaSorter] = None
        self.target_image_path: Optional[str] = None
        self.target_profile: Optional[Dict[str, Any]] = None
        self.media_sources: List[str] = []
        self.discovered_items: List[MediaItem] = []
        self.is_scanning = False

        # Build UI
        self._build_ui()

        # Asynchronously initialize the engine models on startup so UI doesn't delay
        self.after(100, self._async_init_engine)

    def _async_init_engine(self):
        def _worker():
            try:
                self.engine = FaceEngine()
                self.sorter = MediaSorter(self.engine)
                self.after(0, lambda: self.status_label.configure(
                    text="✓ AI Models Loaded & Ready. Please select a reference photo.",
                    text_color="#4ade80"
                ))
            except Exception as e:
                self.after(0, lambda: self.status_label.configure(
                    text=f"⚠ Error loading models: {e}",
                    text_color="#f87171"
                ))
        threading.Thread(target=_worker, daemon=True).start()

    def _build_ui(self):
        # Grid layout (1 row, 2 main columns: left config panel, right results panel)
        self.grid_columnconfigure(0, weight=0, minsize=400)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # ==========================================
        # LEFT PANEL: Controls & Configuration
        # ==========================================
        left_frame = ctk.CTkScrollableFrame(self, width=410, corner_radius=12)
        left_frame.grid(row=0, column=0, padx=(16, 8), pady=16, sticky="nsew")

        # Title / Header
        title_label = ctk.CTkLabel(
            left_frame,
            text="👤 Face Media Filter",
            font=ctk.CTkFont(size=22, weight="bold")
        )
        title_label.pack(anchor="w", padx=10, pady=(10, 4))

        subtitle_label = ctk.CTkLabel(
            left_frame,
            text="Filter videos & photos by person presence",
            font=ctk.CTkFont(size=12),
            text_color="gray70"
        )
        subtitle_label.pack(anchor="w", padx=10, pady=(0, 14))

        # --- SECTION 1: Reference Photo ---
        sec1_frame = ctk.CTkFrame(left_frame, corner_radius=10)
        sec1_frame.pack(fill="x", padx=6, pady=6)

        ctk.CTkLabel(
            sec1_frame,
            text="1. Target Reference Photo",
            font=ctk.CTkFont(size=14, weight="bold")
        ).pack(anchor="w", padx=12, pady=(10, 4))

        ref_btn_frame = ctk.CTkFrame(sec1_frame, fg_color="transparent")
        ref_btn_frame.pack(fill="x", padx=12, pady=4)

        self.btn_select_target = ctk.CTkButton(
            ref_btn_frame,
            text="Select Reference Photo...",
            command=self._on_select_target_photo,
            height=32
        )
        self.btn_select_target.pack(side="left", expand=True, fill="x", padx=(0, 4))

        # Face Preview Box
        self.preview_frame = ctk.CTkFrame(sec1_frame, height=120, corner_radius=8, fg_color="#1e293b")
        self.preview_frame.pack(fill="x", padx=12, pady=(6, 8))
        self.preview_frame.pack_propagate(False)

        self.face_preview_label = ctk.CTkLabel(
            self.preview_frame,
            text="No reference photo selected\n(Click 'Select Reference Photo')",
            font=ctk.CTkFont(size=11),
            text_color="gray60"
        )
        self.face_preview_label.pack(expand=True)

        self.target_status_badge = ctk.CTkLabel(
            sec1_frame,
            text="Awaiting photo selection",
            font=ctk.CTkFont(size=11),
            text_color="gray60"
        )
        self.target_status_badge.pack(anchor="w", padx=12, pady=(0, 10))

        # --- SECTION 2: Media Source Files ---
        sec2_frame = ctk.CTkFrame(left_frame, corner_radius=10)
        sec2_frame.pack(fill="x", padx=6, pady=6)

        ctk.CTkLabel(
            sec2_frame,
            text="2. Photos & Videos to Filter",
            font=ctk.CTkFont(size=14, weight="bold")
        ).pack(anchor="w", padx=12, pady=(10, 4))

        media_btn_row = ctk.CTkFrame(sec2_frame, fg_color="transparent")
        media_btn_row.pack(fill="x", padx=12, pady=4)

        self.btn_select_folder = ctk.CTkButton(
            media_btn_row,
            text="📁 Select Folder",
            command=self._on_select_media_folder,
            height=30
        )
        self.btn_select_folder.pack(side="left", expand=True, fill="x", padx=(0, 4))

        self.btn_select_files = ctk.CTkButton(
            media_btn_row,
            text="📄 Select Files",
            command=self._on_select_media_files,
            height=30,
            fg_color="#475569",
            hover_color="#334155"
        )
        self.btn_select_files.pack(side="left", expand=True, fill="x", padx=(4, 0))

        self.var_recursive = ctk.BooleanVar(value=True)
        self.chk_recursive = ctk.CTkCheckBox(
            sec2_frame,
            text="Include subfolders recursively",
            variable=self.var_recursive,
            font=ctk.CTkFont(size=12)
        )
        self.chk_recursive.pack(anchor="w", padx=12, pady=(6, 4))

        self.media_count_label = ctk.CTkLabel(
            sec2_frame,
            text="No media selected yet",
            font=ctk.CTkFont(size=11),
            text_color="gray60"
        )
        self.media_count_label.pack(anchor="w", padx=12, pady=(0, 10))

        # --- SECTION 3: Decision Mode & Action ---
        sec3_frame = ctk.CTkFrame(left_frame, corner_radius=10)
        sec3_frame.pack(fill="x", padx=6, pady=6)

        ctk.CTkLabel(
            sec3_frame,
            text="3. Filter Logic (Action Mode)",
            font=ctk.CTkFont(size=14, weight="bold")
        ).pack(anchor="w", padx=12, pady=(10, 4))

        self.var_mode = ctk.StringVar(value="keep_match")

        self.rad_keep = ctk.CTkRadioButton(
            sec3_frame,
            text="KEEP if target is in photo/video\n(Delete files without target)",
            variable=self.var_mode,
            value="keep_match",
            font=ctk.CTkFont(size=12)
        )
        self.rad_keep.pack(anchor="w", padx=12, pady=(6, 4))

        self.rad_delete = ctk.CTkRadioButton(
            sec3_frame,
            text="DELETE if target is in photo/video\n(Keep files without target)",
            variable=self.var_mode,
            value="delete_match",
            font=ctk.CTkFont(size=12)
        )
        self.rad_delete.pack(anchor="w", padx=12, pady=(4, 10))

        # Action destination
        ctk.CTkLabel(
            sec3_frame,
            text="How to handle deleted files:",
            font=ctk.CTkFont(size=12, weight="bold")
        ).pack(anchor="w", padx=12, pady=(4, 2))

        self.var_action = ctk.StringVar(value="trash")
        self.opt_action = ctk.CTkOptionMenu(
            sec3_frame,
            variable=self.var_action,
            values=[
                "Send to Recycle Bin (Safe / Recoverable)",
                "Move to a Separate Folder...",
                "Permanently Delete from Disk",
                "Preview Only (Simulate / Dry Run)"
            ],
            command=self._on_action_type_changed
        )
        self.opt_action.pack(fill="x", padx=12, pady=(2, 6))

        self.custom_folder_frame = ctk.CTkFrame(sec3_frame, fg_color="transparent")
        self.custom_dest_path: Optional[str] = None
        self.lbl_dest_folder = ctk.CTkLabel(
            self.custom_folder_frame,
            text="No destination folder selected",
            font=ctk.CTkFont(size=11),
            text_color="gray70"
        )
        self.btn_dest_folder = ctk.CTkButton(
            self.custom_folder_frame,
            text="Choose Output Folder...",
            command=self._on_select_dest_folder,
            height=28
        )

        # --- SECTION 4: Sensitivity & Video Settings ---
        sec4_frame = ctk.CTkFrame(left_frame, corner_radius=10)
        sec4_frame.pack(fill="x", padx=6, pady=6)

        ctk.CTkLabel(
            sec4_frame,
            text="4. Tuning & Sensitivity",
            font=ctk.CTkFont(size=14, weight="bold")
        ).pack(anchor="w", padx=12, pady=(10, 4))

        # Threshold slider
        thresh_label_row = ctk.CTkFrame(sec4_frame, fg_color="transparent")
        thresh_label_row.pack(fill="x", padx=12, pady=(4, 0))
        ctk.CTkLabel(thresh_label_row, text="Match Strictness:", font=ctk.CTkFont(size=12)).pack(side="left")
        self.lbl_thresh_val = ctk.CTkLabel(
            thresh_label_row,
            text=f"{DEFAULT_COSINE_THRESHOLD:.2f} (Balanced)",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color="#60a5fa"
        )
        self.lbl_thresh_val.pack(side="right")

        self.slider_thresh = ctk.CTkSlider(
            sec4_frame,
            from_=0.25,
            to=0.65,
            number_of_steps=40,
            command=self._on_thresh_slider_changed
        )
        self.slider_thresh.set(DEFAULT_COSINE_THRESHOLD)
        self.slider_thresh.pack(fill="x", padx=12, pady=(2, 6))

        # Video sampling slider
        vid_label_row = ctk.CTkFrame(sec4_frame, fg_color="transparent")
        vid_label_row.pack(fill="x", padx=12, pady=(4, 0))
        ctk.CTkLabel(vid_label_row, text="Video Frame Interval:", font=ctk.CTkFont(size=12)).pack(side="left")
        self.lbl_vid_val = ctk.CTkLabel(
            vid_label_row,
            text="1.0s (Recommended)",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color="#60a5fa"
        )
        self.lbl_vid_val.pack(side="right")

        self.slider_video = ctk.CTkSlider(
            sec4_frame,
            from_=0.2,
            to=4.0,
            number_of_steps=38,
            command=self._on_video_slider_changed
        )
        self.slider_video.set(1.0)
        self.slider_video.pack(fill="x", padx=12, pady=(2, 10))

        # Start / Stop Scan Buttons
        self.btn_start_scan = ctk.CTkButton(
            left_frame,
            text="🔍 Start Scan & Identify",
            font=ctk.CTkFont(size=15, weight="bold"),
            height=42,
            fg_color="#2563eb",
            hover_color="#1d4ed8",
            command=self._on_start_scan_clicked
        )
        self.btn_start_scan.pack(fill="x", padx=6, pady=(12, 4))

        self.btn_cancel_scan = ctk.CTkButton(
            left_frame,
            text="■ Stop Scan",
            font=ctk.CTkFont(size=13),
            height=32,
            fg_color="#ef4444",
            hover_color="#dc2626",
            command=self._on_cancel_scan_clicked,
            state="disabled"
        )
        self.btn_cancel_scan.pack(fill="x", padx=6, pady=(4, 12))

        # ==========================================
        # RIGHT PANEL: Progress, Stats & Results
        # ==========================================
        right_frame = ctk.CTkFrame(self, corner_radius=12)
        right_frame.grid(row=0, column=1, padx=(8, 16), pady=16, sticky="nsew")
        right_frame.grid_rowconfigure(2, weight=1)
        right_frame.grid_columnconfigure(0, weight=1)

        # 1. Top Status & Progress Bar
        progress_box = ctk.CTkFrame(right_frame, corner_radius=10, fg_color="#1e293b")
        progress_box.grid(row=0, column=0, padx=12, pady=12, sticky="ew")

        self.status_label = ctk.CTkLabel(
            progress_box,
            text="Loading AI models...",
            font=ctk.CTkFont(size=13, weight="bold"),
            text_color="#94a3b8"
        )
        self.status_label.pack(anchor="w", padx=14, pady=(10, 4))

        self.progress_bar = ctk.CTkProgressBar(progress_box)
        self.progress_bar.set(0.0)
        self.progress_bar.pack(fill="x", padx=14, pady=(2, 6))

        self.progress_detail_label = ctk.CTkLabel(
            progress_box,
            text="0 / 0 files scanned",
            font=ctk.CTkFont(size=11),
            text_color="gray65"
        )
        self.progress_detail_label.pack(anchor="w", padx=14, pady=(0, 10))

        # 2. Stats Dashboard (Cards)
        stats_frame = ctk.CTkFrame(right_frame, fg_color="transparent")
        stats_frame.grid(row=1, column=0, padx=12, pady=(0, 8), sticky="ew")
        for i in range(4):
            stats_frame.grid_columnconfigure(i, weight=1)

        self.card_total = self._create_stat_card(stats_frame, 0, "Total Files", "0", "#64748b")
        self.card_matched = self._create_stat_card(stats_frame, 1, "Target Found", "0", "#3b82f6")
        self.card_keep = self._create_stat_card(stats_frame, 2, "To KEEP", "0", "#10b981")
        self.card_delete = self._create_stat_card(stats_frame, 3, "To DELETE", "0", "#ef4444")

        # 3. Results Table / List
        results_header = ctk.CTkFrame(right_frame, height=36, corner_radius=6, fg_color="#334155")
        results_header.grid(row=2, column=0, padx=12, pady=(4, 0), sticky="ew")
        results_header.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(results_header, text="Type", width=50, font=ctk.CTkFont(size=12, weight="bold")).grid(row=0, column=0, padx=6)
        ctk.CTkLabel(results_header, text="Filename & Details", anchor="w", font=ctk.CTkFont(size=12, weight="bold")).grid(row=0, column=1, sticky="w", padx=6)
        ctk.CTkLabel(results_header, text="Match Score", width=90, font=ctk.CTkFont(size=12, weight="bold")).grid(row=0, column=2, padx=6)
        ctk.CTkLabel(results_header, text="Planned Action", width=120, font=ctk.CTkFont(size=12, weight="bold")).grid(row=0, column=3, padx=6)
        ctk.CTkLabel(results_header, text="Action Status", width=130, font=ctk.CTkFont(size=12, weight="bold")).grid(row=0, column=4, padx=6)

        self.results_scroll = ctk.CTkScrollableFrame(right_frame, corner_radius=8)
        self.results_scroll.grid(row=3, column=0, padx=12, pady=(2, 10), sticky="nsew")
        right_frame.grid_rowconfigure(3, weight=1)

        # Empty state label
        self.empty_label = ctk.CTkLabel(
            self.results_scroll,
            text="No scan results to display yet.\nSelect a reference photo and files/folder, then click 'Start Scan & Identify'.",
            font=ctk.CTkFont(size=13),
            text_color="gray50"
        )
        self.empty_label.pack(pady=60)

        # 4. Bottom Action Bar
        bottom_bar = ctk.CTkFrame(right_frame, fg_color="transparent")
        bottom_bar.grid(row=4, column=0, padx=12, pady=(4, 12), sticky="ew")

        self.btn_execute = ctk.CTkButton(
            bottom_bar,
            text="⚡ Apply & Execute File Actions",
            font=ctk.CTkFont(size=14, weight="bold"),
            height=40,
            fg_color="#10b981",
            hover_color="#059669",
            command=self._on_execute_actions_clicked,
            state="disabled"
        )
        self.btn_execute.pack(side="left", padx=(0, 10))

        self.btn_export_csv = ctk.CTkButton(
            bottom_bar,
            text="📊 Export CSV Report",
            font=ctk.CTkFont(size=13),
            height=36,
            fg_color="#475569",
            hover_color="#334155",
            command=self._on_export_csv_clicked,
            state="disabled"
        )
        self.btn_export_csv.pack(side="left", padx=4)

    def _create_stat_card(self, parent, col, title, value, color) -> ctk.CTkLabel:
        box = ctk.CTkFrame(parent, corner_radius=8, fg_color="#1e293b")
        box.grid(row=0, column=col, padx=4, pady=4, sticky="ew")

        ctk.CTkLabel(
            box,
            text=title,
            font=ctk.CTkFont(size=11),
            text_color="gray70"
        ).pack(anchor="w", padx=10, pady=(6, 0))

        val_label = ctk.CTkLabel(
            box,
            text=value,
            font=ctk.CTkFont(size=20, weight="bold"),
            text_color=color
        )
        val_label.pack(anchor="w", padx=10, pady=(0, 6))
        return val_label

    def _on_thresh_slider_changed(self, value):
        val = float(value)
        if val < 0.35:
            desc = "Lenient (More Matches)"
        elif val > 0.45:
            desc = "Strict (Few False Positives)"
        else:
            desc = "Balanced"
        self.lbl_thresh_val.configure(text=f"{val:.2f} ({desc})")

    def _on_video_slider_changed(self, value):
        val = float(value)
        desc = "Thorough" if val < 0.8 else ("Fast" if val > 2.0 else "Balanced")
        self.lbl_vid_val.configure(text=f"{val:.1f}s ({desc})")

    def _on_action_type_changed(self, choice):
        if "Separate Folder" in choice:
            self.custom_folder_frame.pack(fill="x", padx=12, pady=(4, 6))
            self.btn_dest_folder.pack(side="left", padx=(0, 4))
            self.lbl_dest_folder.pack(side="left", fill="x", expand=True)
        else:
            self.custom_folder_frame.pack_forget()

    def _on_select_dest_folder(self):
        folder = filedialog.askdirectory(title="Select Destination Folder for Files")
        if folder:
            self.custom_dest_path = folder
            self.lbl_dest_folder.configure(text=os.path.basename(folder))

    def _on_select_target_photo(self):
        file_path = filedialog.askopenfilename(
            title="Select Target Person Reference Photo",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.webp *.bmp *.tiff *.tif"),
                ("All files", "*.*")
            ]
        )
        if not file_path:
            return

        self.target_image_path = file_path
        self.target_status_badge.configure(text="Analyzing photo for human face...", text_color="#facc15")

        def _worker():
            if not self.sorter:
                self.sorter = MediaSorter()
            profile = self.sorter.load_target_picture(file_path)
            self.target_profile = profile
            self.after(0, lambda: self._update_target_preview(profile))

        threading.Thread(target=_worker, daemon=True).start()

    def _update_target_preview(self, profile: Dict[str, Any]):
        if profile.get("error"):
            self.target_status_badge.configure(
                text=f"⚠ {profile['error']}",
                text_color="#ef4444"
            )
            self.face_preview_label.configure(
                text="No face detected in reference photo.\nPlease choose a clearer picture.",
                image=None
            )
            return

        count = profile["face_count"]
        self.target_status_badge.configure(
            text=f"✓ Face detected successfully ({count} face{'s' if count > 1 else ''} found)",
            text_color="#10b981"
        )

        crop_rgb = profile.get("face_crop")
        if crop_rgb is not None:
            pil_img = Image.fromarray(crop_rgb)
            # Scale thumbnail to fit 100x100
            pil_img.thumbnail((100, 100))
            ctk_img = ctk.CTkImage(light_image=pil_img, dark_image=pil_img, size=pil_img.size)
            self.face_preview_label.configure(text="", image=ctk_img)
            self.face_preview_label.image = ctk_img

    def _on_select_media_folder(self):
        folder = filedialog.askdirectory(title="Select Media Folder to Scan")
        if not folder:
            return
        self.media_sources = [folder]
        self._refresh_discovered_count()

    def _on_select_media_files(self):
        files = filedialog.askopenfilenames(
            title="Select Photos & Videos",
            filetypes=[
                ("Media files", "*.jpg *.jpeg *.png *.webp *.bmp *.mp4 *.mkv *.avi *.mov *.wmv *.webm"),
                ("Images", "*.jpg *.jpeg *.png *.webp *.bmp"),
                ("Videos", "*.mp4 *.mkv *.avi *.mov *.wmv *.webm"),
                ("All files", "*.*")
            ]
        )
        if not files:
            return
        self.media_sources = list(files)
        self._refresh_discovered_count()

    def _refresh_discovered_count(self):
        if not self.media_sources:
            return
        items = MediaSorter.discover_media_files(self.media_sources, recursive=self.var_recursive.get())
        self.discovered_items = items
        img_c = sum(1 for it in items if it.file_type == "image")
        vid_c = sum(1 for it in items if it.file_type == "video")
        self.media_count_label.configure(
            text=f"Found {len(items)} files ({img_c} photos, {vid_c} videos)",
            text_color="#60a5fa"
        )
        self.card_total.configure(text=str(len(items)))

    def _on_cancel_scan_clicked(self):
        if self.sorter:
            self.sorter.cancel_scan()
        self.status_label.configure(text="Cancelling scan...", text_color="#facc15")

    def _on_start_scan_clicked(self):
        if not self.target_profile or self.target_profile.get("error"):
            messagebox.showwarning("Reference Photo Required", "Please select a reference photo containing a face first.")
            return

        if not self.discovered_items:
            self._refresh_discovered_count()

        if not self.discovered_items:
            messagebox.showwarning("No Media Selected", "Please select a media folder or media files to scan.")
            return

        self.is_scanning = True
        self.btn_start_scan.configure(state="disabled")
        self.btn_cancel_scan.configure(state="normal")
        self.btn_execute.configure(state="disabled")
        self.btn_export_csv.configure(state="disabled")

        # Clear existing result rows
        for widget in self.results_scroll.winfo_children():
            widget.destroy()

        threshold = self.slider_thresh.get()
        video_sample = self.slider_video.get()
        mode_val = self.var_mode.get()
        filter_mode = FilterMode.KEEP_MATCH if mode_val == "keep_match" else FilterMode.DELETE_MATCH

        # Background worker thread
        def _worker():
            total = len(self.discovered_items)

            def _progress(idx, tot, item, msg):
                self.after(0, lambda: self._update_scan_progress(idx, tot, item, msg))

            self.sorter.scan_media_items(
                items=self.discovered_items,
                filter_mode=filter_mode,
                threshold=threshold,
                video_sample_sec=video_sample,
                progress_callback=_progress
            )
            self.after(0, self._on_scan_completed)

        threading.Thread(target=_worker, daemon=True).start()

    def _update_scan_progress(self, idx, total, item: MediaItem, msg: str):
        pct = (idx / total) if total > 0 else 0.0
        self.progress_bar.set(pct)
        self.status_label.configure(text=msg, text_color="#94a3b8")
        self.progress_detail_label.configure(text=f"{idx} / {total} files scanned ({(pct*100):.1f}%)")

        # Update stats
        matched = sum(1 for it in self.discovered_items[:idx] if it.has_match)
        kept = sum(1 for it in self.discovered_items[:idx] if it.effective_decision == "KEEP")
        deleted = sum(1 for it in self.discovered_items[:idx] if it.effective_decision == "DELETE")

        self.card_matched.configure(text=str(matched))
        self.card_keep.configure(text=str(kept))
        self.card_delete.configure(text=str(deleted))

    def _on_scan_completed(self):
        self.is_scanning = False
        self.btn_start_scan.configure(state="normal")
        self.btn_cancel_scan.configure(state="disabled")
        self.btn_execute.configure(state="normal")
        self.btn_export_csv.configure(state="normal")

        self.status_label.configure(text="✓ Scan Complete! Review results below.", text_color="#10b981")
        self.progress_bar.set(1.0)

        # Render rows into scrollable area
        self._populate_results_table()

    def _populate_results_table(self):
        for widget in self.results_scroll.winfo_children():
            widget.destroy()

        if not self.discovered_items:
            ctk.CTkLabel(self.results_scroll, text="No items found.").pack(pady=40)
            return

        for idx, item in enumerate(self.discovered_items):
            self._create_result_row(item, idx)

    def _create_result_row(self, item: MediaItem, idx: int):
        row_bg = "#1e293b" if idx % 2 == 0 else "#0f172a"
        row_frame = ctk.CTkFrame(self.results_scroll, corner_radius=6, fg_color=row_bg, height=44)
        row_frame.pack(fill="x", pady=2)
        row_frame.pack_propagate(False)
        row_frame.grid_columnconfigure(1, weight=1)

        # Type Icon
        type_icon = "🖼 Image" if item.file_type == "image" else "🎥 Video"
        ctk.CTkLabel(
            row_frame,
            text=type_icon,
            width=65,
            font=ctk.CTkFont(size=11),
            text_color="gray70"
        ).grid(row=0, column=0, padx=6)

        # Filename & Details
        title_box = ctk.CTkFrame(row_frame, fg_color="transparent")
        title_box.grid(row=0, column=1, sticky="w", padx=6)

        ctk.CTkLabel(
            title_box,
            text=item.filename,
            font=ctk.CTkFont(size=12, weight="bold"),
            anchor="w"
        ).pack(anchor="w")

        ctk.CTkLabel(
            title_box,
            text=f"{item.formatted_size} • {item.details}",
            font=ctk.CTkFont(size=10),
            text_color="gray60",
            anchor="w"
        ).pack(anchor="w")

        # Match Score
        score_text = f"{(item.similarity_score * 100):.1f}%" if item.similarity_score > 0 else "0.0%"
        score_color = "#3b82f6" if item.has_match else "gray60"
        ctk.CTkLabel(
            row_frame,
            text=score_text,
            width=90,
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color=score_color
        ).grid(row=0, column=2, padx=6)

        # Planned Decision Badge & Toggle
        dec_color = "#10b981" if item.effective_decision == "KEEP" else "#ef4444"
        btn_decision = ctk.CTkButton(
            row_frame,
            text=f"● {item.effective_decision}",
            width=110,
            height=26,
            fg_color=dec_color,
            hover_color="#059669" if item.effective_decision == "KEEP" else "#dc2626",
            font=ctk.CTkFont(size=11, weight="bold")
        )
        btn_decision.grid(row=0, column=3, padx=6)

        def _toggle_decision():
            new_decision = "DELETE" if item.effective_decision == "KEEP" else "KEEP"
            item.override_decision = new_decision
            new_color = "#10b981" if new_decision == "KEEP" else "#ef4444"
            btn_decision.configure(
                text=f"● {new_decision} (Manual)",
                fg_color=new_color
            )
            # Update stats
            kept = sum(1 for it in self.discovered_items if it.effective_decision == "KEEP")
            deleted = sum(1 for it in self.discovered_items if it.effective_decision == "DELETE")
            self.card_keep.configure(text=str(kept))
            self.card_delete.configure(text=str(deleted))

        btn_decision.configure(command=_toggle_decision)

        # Action status label
        lbl_status = ctk.CTkLabel(
            row_frame,
            text=item.action_status,
            width=130,
            font=ctk.CTkFont(size=11),
            text_color="gray75"
        )
        lbl_status.grid(row=0, column=4, padx=6)
        item._lbl_status = lbl_status

    def _on_export_csv_clicked(self):
        if not self.discovered_items:
            return
        save_path = filedialog.asksaveasfilename(
            title="Save CSV Report",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if save_path:
            MediaSorter.export_report_csv(self.discovered_items, save_path)
            messagebox.showinfo("Export Successful", f"Saved scan report to:\n{save_path}")

    def _on_execute_actions_clicked(self):
        items_to_delete = [it for it in self.discovered_items if it.effective_decision == "DELETE"]
        if not items_to_delete:
            messagebox.showinfo("No Files Marked for Deletion", "All scanned files are marked as KEEP. Nothing to delete.")
            return

        action_selection = self.var_action.get()
        if "Recycle Bin" in action_selection:
            action_type = ActionType.TRASH
            action_desc = f"move {len(items_to_delete)} files to the Windows Recycle Bin"
        elif "Separate Folder" in action_selection:
            if not self.custom_dest_path:
                messagebox.showwarning("Destination Folder Missing", "Please select a destination folder before applying.")
                return
            action_type = ActionType.MOVE_FOLDER
            action_desc = f"move {len(items_to_delete)} files to '{self.custom_dest_path}'"
        elif "Permanently Delete" in action_selection:
            action_type = ActionType.PERMANENT_DELETE
            action_desc = f"PERMANENTLY DELETE {len(items_to_delete)} files from your disk (irreversible)"
        else:
            action_type = ActionType.SIMULATE
            action_desc = f"simulate actions on {len(items_to_delete)} files (dry run)"

        # Confirmation Dialog
        confirmed = messagebox.askyesno(
            "Confirm Action",
            f"Are you sure you want to {action_desc}?\n\nFiles kept: {len(self.discovered_items) - len(items_to_delete)}\nFiles to remove/move: {len(items_to_delete)}",
            icon="warning" if action_type == ActionType.PERMANENT_DELETE else "question"
        )
        if not confirmed:
            return

        self.btn_execute.configure(state="disabled")
        self.status_label.configure(text="Executing file operations...", text_color="#facc15")

        def _worker():
            stats = self.sorter.execute_actions(
                items=self.discovered_items,
                action_type=action_type,
                destination_dir=self.custom_dest_path
            )
            self.after(0, lambda: self._on_actions_completed(stats))

        threading.Thread(target=_worker, daemon=True).start()

    def _on_actions_completed(self, stats: Dict[str, Any]):
        self.btn_execute.configure(state="normal")
        self.status_label.configure(
            text=f"✓ Complete! Kept: {stats['kept']}, Removed: {stats['deleted']}, Errors: {stats['errors']}",
            text_color="#10b981"
        )

        # Refresh row labels
        for item in self.discovered_items:
            if hasattr(item, "_lbl_status"):
                item._lbl_status.configure(text=item.action_status)

        messagebox.showinfo(
            "Operation Complete",
            f"Execution finished!\n\n"
            f"• Kept: {stats['kept']} files\n"
            f"• Processed: {stats['deleted']} files\n"
            f"• Errors: {stats['errors']}"
        )


def launch_gui():
    app = FaceMediaFilterApp()
    app.mainloop()


if __name__ == "__main__":
    launch_gui()
