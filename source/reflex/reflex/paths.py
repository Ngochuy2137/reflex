"""Default locations of downloaded / generated data.

Everything lives under ``$REFLEX_DATA_DIR`` (default: ``<repo>/data``), populated by
``scripts/download_checkpoints.sh`` and ``tools/motion/*``.
"""
import os

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
DATA_DIR = os.environ.get("REFLEX_DATA_DIR", os.path.join(REPO_ROOT, "data"))

MOTION_FILE = os.path.join(DATA_DIR, "motions", "walk_then_catch_v17_30s.npz")
AMP_DATASET = os.path.join(DATA_DIR, "amp", "amp_dataset.pt")
READY_POSE = os.path.join(REPO_ROOT, "assets", "motions", "ready_pose_capture.npy")
CHECKPOINT_DIR = os.path.join(DATA_DIR, "checkpoints")
