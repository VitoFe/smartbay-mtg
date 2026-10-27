#!/usr/bin/env python
"""smartbay - one entry point for the SmartBayDO pipeline. See `python smartbay.py --help`."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.pipeline.main import main

if __name__ == '__main__':
    raise SystemExit(main())
