"""
cli.py
Command-Line Interface for Face Media Filter.
Supports scanning folders/files of photos and videos, matching faces,
and keeping or deleting based on the selected mode.
"""

import os
import sys
import argparse
from pathlib import Path

# Ensure UTF-8 output on Windows terminals
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

from colorama import init, Fore, Style

from face_engine import FaceEngine, DEFAULT_COSINE_THRESHOLD
from sorter_logic import MediaSorter, FilterMode, ActionType

init(autoreset=True)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Filter and organize photos and videos based on facial recognition."
    )
    parser.add_argument(
        "--target", "-t", required=True,
        help="Path to the reference photo containing the target person."
    )
    parser.add_argument(
        "--media-dir", "-d",
        help="Directory containing photos and videos to scan."
    )
    parser.add_argument(
        "--files", "-f", nargs="*", default=[],
        help="Specific photo and video files to scan."
    )
    parser.add_argument(
        "--mode", "-m", choices=["keep_match", "delete_match"], default="keep_match",
        help=(
            "'keep_match': KEEP file if person is present, DELETE otherwise (default). "
            "'delete_match': DELETE file if person is present, KEEP otherwise."
        )
    )
    parser.add_argument(
        "--action", "-a", choices=["trash", "move", "permanent", "simulate"], default="trash",
        help=(
            "'trash': Move deleted files to Windows Recycle Bin (Safe, default). "
            "'move': Move to a specified destination directory without deleting. "
            "'permanent': Permanently delete files from disk (Use with caution!). "
            "'simulate': Dry-run preview without modifying any files."
        )
    )
    parser.add_argument(
        "--dest-dir",
        help="Destination folder when using --action move."
    )
    parser.add_argument(
        "--threshold", type=float, default=DEFAULT_COSINE_THRESHOLD,
        help=f"Face recognition cosine similarity threshold (default: {DEFAULT_COSINE_THRESHOLD}). Higher = stricter."
    )
    parser.add_argument(
        "--video-interval", type=float, default=1.0,
        help="Seconds between sampled frames in videos (default: 1.0s). Lower = more thorough, higher = faster."
    )
    parser.add_argument(
        "--no-recursive", action="store_true",
        help="Do not scan subdirectories recursively."
    )
    parser.add_argument(
        "--report", "-r",
        help="Path to save detailed scan results as a CSV report."
    )
    parser.add_argument(
        "--yes", "-y", action="store_true",
        help="Skip confirmation prompt before executing file operations."
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # 1. Validate inputs
    target_path = os.path.abspath(args.target)
    if not os.path.isfile(target_path):
        print(f"{Fore.RED}Error: Target image file not found: {target_path}")
        sys.exit(1)

    sources = []
    if args.media_dir:
        media_dir = os.path.abspath(args.media_dir)
        if not os.path.isdir(media_dir):
            print(f"{Fore.RED}Error: Media directory not found: {media_dir}")
            sys.exit(1)
        sources.append(media_dir)

    for f in args.files:
        fp = os.path.abspath(f)
        if os.path.isfile(fp):
            sources.append(fp)
        else:
            print(f"{Fore.YELLOW}Warning: File not found, skipping: {fp}")

    if not sources:
        print(f"{Fore.RED}Error: No media sources provided. Use --media-dir or --files.")
        sys.exit(1)

    if args.action == "move" and not args.dest_dir:
        print(f"{Fore.RED}Error: --dest-dir is required when --action is set to 'move'.")
        sys.exit(1)

    filter_mode = FilterMode(args.mode)
    action_type = ActionType(args.action)

    print(f"\n{Fore.CYAN}==================================================")
    print(f"{Fore.CYAN}       FACE MEDIA SORTER & FILTER CLI             ")
    print(f"{Fore.CYAN}==================================================")
    print(f"Target Reference: {Style.BRIGHT}{os.path.basename(target_path)}")
    print(f"Mode:             {Fore.YELLOW}{'KEEP files with target person (delete others)' if filter_mode == FilterMode.KEEP_MATCH else 'DELETE files with target person (keep others)'}")
    print(f"Action:           {Fore.MAGENTA}{action_type.value.upper()}")
    print(f"Threshold:        {args.threshold:.2f}")
    print(f"Video Sampling:   Every {args.video_interval:.1f}s")
    print(f"==================================================\n")

    # 2. Initialize Engine & Load Target
    print(f"{Fore.WHITE}Initializing face detection & recognition models...")
    sorter = MediaSorter()

    print(f"{Fore.WHITE}Analyzing reference photo: {target_path}...")
    profile = sorter.load_target_picture(target_path)
    if profile.get("error"):
        print(f"{Fore.RED}Error analyzing reference picture: {profile['error']}")
        sys.exit(1)

    print(f"{Fore.GREEN}✓ Face successfully detected in reference picture! (Total faces found: {profile['face_count']})")

    # 3. Discover Media
    print(f"\n{Fore.WHITE}Discovering media files in sources...")
    items = sorter.discover_media_files(sources, recursive=not args.no_recursive)
    if not items:
        print(f"{Fore.YELLOW}No supported photos or videos found in the specified locations.")
        sys.exit(0)

    img_count = sum(1 for it in items if it.file_type == "image")
    vid_count = sum(1 for it in items if it.file_type == "video")
    print(f"{Fore.GREEN}✓ Discovered {len(items)} media files ({img_count} photos, {vid_count} videos).")

    # 4. Scan Media
    print(f"\n{Fore.CYAN}--- Starting Media Scan ---")

    def progress_callback(idx, total, item, msg):
        pct = (idx / total) * 100 if total > 0 else 0
        match_symbol = "✓ FOUND" if item.has_match else "✗ NO"
        print(f"\r[{idx}/{total}] ({pct:5.1f}%) {item.filename[:30]:<30} | Match: {match_symbol} ({item.similarity_score:.2f}) -> Decision: {item.decision:<6}", end="", flush=True)

    items = sorter.scan_media_items(
        items=items,
        filter_mode=filter_mode,
        threshold=args.threshold,
        video_sample_sec=args.video_interval,
        progress_callback=progress_callback
    )
    print("\n")

    # 5. Display Summary
    matched_count = sum(1 for it in items if it.has_match)
    keep_count = sum(1 for it in items if it.decision == "KEEP")
    del_count = sum(1 for it in items if it.decision == "DELETE")

    print(f"{Fore.CYAN}==================================================")
    print(f"{Fore.CYAN}                 SCAN SUMMARY                     ")
    print(f"{Fore.CYAN}==================================================")
    print(f"Total Files Scanned: {len(items)}")
    print(f"Target Person Found: {Fore.GREEN}{matched_count} files")
    print(f"Target Person Not In:{Fore.YELLOW}{len(items) - matched_count} files")
    print(f"Files to KEEP:       {Fore.GREEN}{keep_count}")
    print(f"Files to DELETE:     {Fore.RED}{del_count}")
    print(f"==================================================")

    # 6. Export report if requested
    if args.report:
        report_path = os.path.abspath(args.report)
        sorter.export_report_csv(items, report_path)
        print(f"{Fore.GREEN}✓ Saved detailed scan report to: {report_path}")

    # 7. Action Execution
    if del_count == 0:
        print(f"\n{Fore.GREEN}No files are marked for deletion. Exiting.")
        return

    if action_type == ActionType.SIMULATE:
        print(f"\n{Fore.YELLOW}Simulation mode active. No files were modified.")
        return

    if not args.yes:
        action_verb = "send to Recycle Bin" if action_type == ActionType.TRASH else (
            f"move to '{args.dest_dir}'" if action_type == ActionType.MOVE_FOLDER else "PERMANENTLY DELETE"
        )
        prompt = f"\nAre you sure you want to {action_verb} {del_count} files? (y/N): "
        resp = input(prompt).strip().lower()
        if resp not in ["y", "yes"]:
            print(f"{Fore.YELLOW}Action cancelled by user.")
            return

    print(f"\n{Fore.CYAN}Executing {action_type.value.upper()} on marked files...")
    stats = sorter.execute_actions(
        items=items,
        action_type=action_type,
        destination_dir=args.dest_dir
    )

    print(f"\n{Fore.GREEN}✓ Operation completed!")
    print(f"  Files Kept:     {stats['kept']}")
    print(f"  Files Deleted:  {stats['deleted']} (Action: {action_type.value})")
    if stats["errors"] > 0:
        print(f"  {Fore.RED}Errors encountered: {stats['errors']}")
        for err in stats["error_details"]:
            print(f"    - {err}")


if __name__ == "__main__":
    main()
