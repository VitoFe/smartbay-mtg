"""CI Acceptance Test: Assert zero fabricated or hardcoded metric literals in src/."""

import re
import os
import sys
import ast

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
src_dir = os.path.join(project_root, "src")

# Matches patterns like "mae": 12.34 or "bottom_mae_umol_l": 15.6 or "l5_coordinate_field_mean_mae": 15.93
FORBIDDEN_METRIC_LITERAL = re.compile(
    r'["\'][a-z_0-9]*(mae|rmse|f1|picp|mpiw|r2|skill|bottom_mae)[a-z_0-9]*["\']\s*:\s*-?\d+\.\d+'
)
FORBIDDEN_METRIC_STRING = re.compile(
    r'\b(?:VoI|MAE|RMSE|PICP|MPIW|recall|precision|F1|R2)\s*=\s*[+-]?(?:\d+(?:\.\d*)?|\.\d+)'
    r'|\bp\s*<\s*0\.001\b', re.IGNORECASE
)


def hardcoded_metric_strings(source):
    """Inspect runtime string values, including constant portions of f-strings.

    Comments and actual Python docstrings are documentation, not emitted results.
    Formatted values such as f'MAE = {mae:.2f}' remain valid.
    """
    tree = ast.parse(source)
    docstrings = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.body and isinstance(node.body[0], ast.Expr):
                value = node.body[0].value
                if isinstance(value, ast.Constant) and isinstance(value.value, str):
                    docstrings.add(id(value))
    return [(node.lineno, node.value) for node in ast.walk(tree)
            if isinstance(node, ast.Constant) and isinstance(node.value, str)
            and id(node) not in docstrings and FORBIDDEN_METRIC_STRING.search(node.value)]


def test_no_fabricated_metric_literals_in_source():
    violations = []

    for root, _, files in os.walk(src_dir):
        for f in files:
            if f.endswith(".py") and not f.startswith("__"):
                filepath = os.path.join(root, f)
                rel_path = os.path.relpath(filepath, project_root)
                with open(filepath, "r", encoding="utf-8") as fp:
                    source = fp.read()
                    violations.extend(f"{rel_path}:{n}: {value}" for n, value in hardcoded_metric_strings(source))
                    for line_num, line in enumerate(source.splitlines(), 1):
                        # Ignore comments and docstrings
                        stripped = line.strip()
                        if stripped.startswith("#") or stripped.startswith("*"):
                            continue
                        if FORBIDDEN_METRIC_LITERAL.search(line):
                            violations.append(f"{rel_path}:{line_num}: {line.strip()}")

    assert not violations, "Found hardcoded metric literals in src/:\n" + "\n".join(violations)


def test_scanner_catches_string_metrics_and_static_p_values():
    assert hardcoded_metric_strings('result = {"claim": "CONFIRMED (VoI = 8.9+ umol/L)"}')
    assert hardcoded_metric_strings('result = f"VoI = {delta:.2f}, p < 0.001"')
    assert not hardcoded_metric_strings('result = f"VoI = {delta:.2f}, p = {p:.6g}"')
    assert not hardcoded_metric_strings('"""Example: MAE = 12.34"""\n# VoI = 8.9')


def test_no_stale_metrics_directory():
    """Asserts that the legacy/stale reports/metrics directory has been completely eradicated."""
    metrics_dir = os.path.join(project_root, "reports", "metrics")
    assert not os.path.exists(metrics_dir), f"Stale directory {metrics_dir} must not exist. All reports must reside in reports/json/."
