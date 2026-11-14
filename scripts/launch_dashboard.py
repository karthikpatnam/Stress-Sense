#!/usr/bin/env python
"""Convenience CLI script to launch the StressSense Interactive Dashboard.

Usage:
    python scripts/launch_dashboard.py [--port 8050] [--no-browser] [--rebuild]
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dashboard.server import main

if __name__ == "__main__":
    main()
