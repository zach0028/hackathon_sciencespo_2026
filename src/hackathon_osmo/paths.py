"""Canonical repository paths (single source of truth across all subpackages)."""

from __future__ import annotations

from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parents[1]

ENV_FILE = PROJECT_ROOT / ".env"
CALIBRATION_FILE = PROJECT_ROOT / "calibration.json"
HILIGHT_JAR = PROJECT_ROOT / "android" / "hilight" / "hilight.jar"
STATIC_DIR = PACKAGE_ROOT / "app" / "static"
LOGS_DIR = PROJECT_ROOT / "logs"
