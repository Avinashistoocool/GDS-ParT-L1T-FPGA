"""Thin wrapper so the hls4ml flow can be run as `python hls/convert.py ...`.
See src/gdsl1/report.py for options."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from gdsl1.report import main  # noqa: E402

if __name__ == "__main__":
    main()
