"""
HuggingFace Spaces entry point.

Launches the Semantic Change Search Engine with the bundled synthetic
DEN fixture (2 AOIs × 8 months, <1 MB). CLIP weights (~1.6 GB) are
downloaded once from HuggingFace into the Space cache on first boot.

To use the full DEN dataset instead, set the environment variable:
    DATASET_ROOT=data/DynamicEarthNet
"""
import os
import sys

root = os.environ.get("DATASET_ROOT", "tests/fixtures/den_tiny")
encoder = os.environ.get("ENCODER", "clip_vitl14")

sys.argv = [
    "app.py",
    # The Space ships only the Dynamic EarthNet fixture, so pin that dataset
    # explicitly (the app's default corpus is now LEVIR-CC, whose data is not on
    # the Space). Other corpora in the dropdown error gracefully here.
    "--dataset", os.environ.get("DATASET", "dynamic_earthnet"),
    "--root", root,
    "--split", "all",
    "--encoder", encoder,
    "--port", "7860",
]

from src.app import main  # noqa: E402

main()
