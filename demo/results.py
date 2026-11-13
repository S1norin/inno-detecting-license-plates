"""Canonical experimental results.

Single source: docs/md/final_results.md (regenerated 2026-11-02 from the
canonical weights). Values are labeled with split and protocol; they are
saved historical measurements, not live computations. EasyOCR rows are a
frozen historical baseline from an earlier environment.
"""

SOURCE = "docs/md/final_results.md (2026-11-02, canonical weights)"

# Detection, test split, 800 original-scale images
DETECTION_TEST = {
    "frozen": {"map50": 0.9073, "precision": 0.9408, "recall": 0.8938, "f1": 0.9167},
    "finetuned": {"map50": 0.9949, "precision": 0.9466, "recall": 0.9975, "f1": 0.9714},
}
DETECTION_VAL = {
    "frozen": {"map50": 0.9245, "precision": 0.9402, "recall": 0.8986, "f1": 0.9189},
    "finetuned": {"map50": 0.9949, "precision": 0.9549, "recall": 0.9986, "f1": 0.9763},
}

# Fine-tuned detector, test, by scope: (mAP@0.5, F1)
DETECTION_FINETUNED_SCOPES = {
    "size": {"small": (0.9950, 0.9647), "medium": (0.9950, 0.9753), "large": (0.9948, 0.9739)},
    "tilt": {"0-9": (0.9947, 0.9707), "10-19": (0.9863, 0.9254), "20+": (0.9950, 0.9769)},
}

# Recognition, test split, 800 samples, mode none, tau 0.5
RECOGNITION_TEST = [
    {"engine": "EasyOCR (frozen baseline)", "protocol": "ocr_only", "char_acc": 0.3474, "exact": 0, "f1": None,
     "note": "historical baseline, earlier environment"},
    {"engine": "EasyOCR (frozen baseline)", "protocol": "end_to_end", "char_acc": 0.3240, "exact": 0, "f1": None,
     "note": "historical baseline, earlier environment"},
    {"engine": "CRNN", "protocol": "ocr_only", "char_acc": 0.7593, "exact": 267, "f1": 0.7271, "note": ""},
    {"engine": "CRNN", "protocol": "end_to_end + frozen YOLO11s", "char_acc": 0.6710, "exact": 148, "f1": 0.5676,
     "note": ""},
    {"engine": "CRNN", "protocol": "end_to_end + fine-tuned YOLO11s", "char_acc": 0.7661, "exact": 296, "f1": 0.7372,
     "note": "final system"},
]

# Preprocessing ablation: validation char accuracy (selection happened on val)
PREPROCESSING_VAL = {"none": 0.7558, "gray_contrast": 0.7469, "canny": 0.4575}

# CRNN test, ocr_only, by scope: (char accuracy, exact rate)
CRNN_OCRONLY_SCOPES = {
    "size": {"small": (0.8790, 0.4751), "medium": (0.8295, 0.4152), "large": (0.5658, 0.1069)},
    "tilt": {"0-9": (0.7996, 0.2335), "10-19": (0.5193, 0.0000), "20+": (0.7740, 0.4000)},
}

# Scale robustness, test split, same weights, no retraining
SCALE_DETECTION = {
    "frozen": {1.0: (0.9073, 0.9271), 0.75: (0.9218, 0.9271), 0.5: (0.9294, 0.9298)},
    "finetuned": {1.0: (0.9949, 0.9714), 0.75: (0.9949, 0.9726), 0.5: (0.9949, 0.9708)},
}
SCALE_RECOGNITION = {
    "ocr_only": {1.0: (0.7593, 267), 0.75: (0.7613, 265), 0.5: (0.7610, 268)},
    "end_to_end + fine-tuned": {1.0: (0.7661, 296), 0.75: (0.7629, 297), 0.5: (0.7555, 279)},
    "end_to_end + frozen": {1.0: (0.6710, 148), 0.75: (0.6713, 157), 0.5: (0.6721, 144)},
}

# Runtime on Apple M3, CPU, single-threaded (historical, not this machine)
RUNTIME_M3 = {
    "frozen": {"detection_median_ms": 77.8, "detection_p95_ms": 88.0},
    "finetuned": {"detection_median_ms": 80.2, "detection_p95_ms": 98.8},
    "crnn_median_ms": 5.7,
    "end_to_end_ms": "83-86",
}

# Documented failure cases (docs/md/final_results.md, Failure analysis)
FAILURE_CASES = [
    {
        "sample_id": "6f3c772ca888fa92_s100",
        "stage": "detection",
        "detector": "frozen",
        "title": "Missed detection (frozen detector)",
        "observation": "Large plate, tilt 19 degrees, zero predictions at any score.",
        "cause": "Weak prior for large plate aspect ratios in the pretrained model; "
                "the 10-19-degree bin is the frozen model's weakest tilt scope.",
        "limitation": "The frozen baseline systematically misses large, tilted plates.",
    },
    {
        "sample_id": "356b4ca635beeca7_s100",
        "stage": "detection",
        "detector": "finetuned",
        "title": "False-positive boxes (fine-tuned detector)",
        "observation": "11 boxes: 1 true (0.84) + 10 lower-confidence false positives.",
        "cause": "Fine-tuning removed the misses but left low-confidence false positives on "
                "cluttered scenes; a higher confidence cutoff would drop them.",
        "limitation": "High recall does not eliminate false positives at low confidence floors.",
    },
    {
        "sample_id": "261f3b723deb67ac_s100",
        "stage": "recognition",
        "detector": None,
        "protocol": "ocr_only",
        "title": "Recognition failure (CRNN, OCR-only)",
        "observation": "GT 京K06350, predicted 皖AF4069; large plate, tilt 29 degrees.",
        "cause": "Full hallucination with low per-character confidence (min 0.24): "
                "large plate glyphs fall outside the training distribution.",
        "limitation": "The recognizer was effectively untrained on large, tilted plates.",
    },
    {
        "sample_id": "588e616d9e7cec6c_s100",
        "stage": "recognition",
        "detector": "finetuned",
        "protocol": "end_to_end",
        "title": "Recognition failure (CRNN, end-to-end)",
        "observation": "GT 苏E050R8, predicted 皖AK81; large plate, tilt 19 degrees.",
        "cause": "The model overfits small-plate statistics: wrong province char and wrong "
                "length on a large tilted plate.",
        "limitation": "Same data gap as OCR-only: large plates are underrepresented in training.",
    },
    {
        "sample_id": "b996022cc9a5ee07_s100",
        "stage": "detection",
        "detector": "frozen",
        "title": "Missed detection, extreme tilt (frozen)",
        "observation": "Large plate, tilt 45 degrees, zero predictions.",
        "cause": "Extreme tilt + large size outside the pretrained distribution.",
        "limitation": "Same blind spot as case 1, at an even more extreme tilt.",
    },
    {
        "sample_id": "bfe070b4b5543916_s100",
        "stage": "recognition",
        "detector": None,
        "protocol": "ocr_only",
        "title": "Recognition failure, tilt bin 10-19 (CRNN)",
        "observation": "GT 浙C7HV30, predicted 皖A9W165; medium plate, tilt 18 degrees.",
        "cause": "Province-char confusion plus plausible-looking digits; "
                "the 10-19-degree tilt bin has 0% exact match on test.",
        "limitation": "The weakest tilt bin for recognition.",
    },
]

DATASET_FACTS = {
    "source": "CCPD2019 (custom selection, not the official benchmark split)",
    "n_originals": 5000,
    "splits": {"train": 3500, "val": 700, "test": 800},
    "scales": [1.0, 0.75, 0.5],
    "n_variants": 15000,
    "seed": 42,
    "bucket_thresholds_px": {"small": "<= 83", "medium": "<= 112", "large": "> 112"},
    "bucket_note": "plate pixel height, derived from the train split only",
}
