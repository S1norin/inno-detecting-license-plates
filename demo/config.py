"""Paths, weight locations, and pinned inference settings for the demo.

Everything here is read-only: the demo never writes into data/, baseline/,
models/, or the weight cache.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = ROOT / "data" / "dataset"
METADATA_CSV = DATASET_DIR / "metadata.csv"

WEIGHTS_DIR = ROOT / "baseline" / "weights"
WEIGHT_FROZEN = WEIGHTS_DIR / "license-plate-finetune-v1s.pt"  # filename per models.json
WEIGHT_FINETUNED = WEIGHTS_DIR / "yolo11_finetune" / "best.pt"
WEIGHT_CRNN = WEIGHTS_DIR / "crnn" / "checkpoint.pt"
EASYOCR_WEIGHTS_DIR = WEIGHTS_DIR / "easyocr"
EASYOCR_MANIFEST = ROOT / "baseline" / "common" / "models.json"

# Pinned SHA-256 digests from baseline/common/models.json
DIGEST_FROZEN = "95e50c25ab7066dd0ca5aec18fa80349676db08697780d1149576461174d2381"
DIGEST_FINETUNED = "0a4f080fe3749b8be145101468a0dd98455cf50117d2acd37575f33e0d261055"
DIGEST_CRNN = "92d1fbeae19a81ddf3ae498a40e4416ed619ad9667736c592ccbb899e59ca2cc"

# Canonical detection settings from baseline/detection/evaluate.py SETTINGS
# and docs/md/final_results.md. Do not change: they define the reported metrics.
YOLO_CANONICAL = {
    "imgsz": 640,
    "iou": 0.7,
    "max_det": 300,
    "half": False,
    "augment": False,
    "rect": False,
}
YOLO_CONF_FOR_AP = 0.001  # confidence floor used for mAP calculation
YOLO_CONF_DEFAULT_UI = 0.25  # display-only default for the interactive UI

CRNN_DEVICE = "cpu"
CRNN_THREADS = 1

MAX_UPLOAD_BYTES = 8 * 1024 * 1024  # 8 MB upload cap
ALLOWED_UPLOAD_TYPES = {"image/jpeg", "image/png", "image/webp"}
