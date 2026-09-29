"""
PitchVision tactical pipeline module.
Exports core analysis functions from football_analyzer.
"""
from football_analyzer import (
    analyze_image,
    analyze_core,
    run_detection,
    calibrate,
    CONFIG,
)

__all__ = ["analyze_image", "analyze_core", "run_detection", "calibrate", "CONFIG"]
