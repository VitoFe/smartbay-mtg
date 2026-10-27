"""Acceptance Test Suite: No Data Synthesis or Magic Number Constant Audit."""

import pytest
import glob
import re
import os


def test_no_synthetic_constants_in_source():
    """
    Grep assertion: scans src/ to ensure no hardcoded physical values
    like o2 = 230.0 or est_seabed = 8.0 + 17.0 are assigned in source files outside config/.
    """
    src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
    py_files = glob.glob(os.path.join(src_dir, "**", "*.py"), recursive=True)

    forbidden_patterns = [
        r"o2\s*=\s*230\.0",
        r"scale_factor\s*=\s*0\.8",
        r"est_seabed\s*=",
        r"o2_fill\s*="
    ]

    violations = []
    for f in py_files:
        with open(f, "r", encoding="utf-8", errors="ignore") as fh:
            content = fh.read()
        for pat in forbidden_patterns:
            if re.search(pat, content):
                violations.append((f, pat))

    assert len(violations) == 0, f"Found forbidden hardcoded data synthesis patterns in source code: {violations}"
