"""
main.py
Main entry point for Face Media Filter & Sorter.
Launches the modern Desktop GUI by default, or runs CLI mode if flags are passed.
"""

import sys
import os

# Add local directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    # If arguments are passed (except --gui), forward to CLI
    if len(sys.argv) > 1 and sys.argv[1] not in ["--gui", "-g"]:
        from cli import main as cli_main
        cli_main()
    else:
        # Launch modern GUI
        from gui import launch_gui
        launch_gui()


if __name__ == "__main__":
    main()
