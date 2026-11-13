"""Drawing helpers for Streamlit: GT boxes, predictions, crops."""

import cv2

COLOR_GT = (0, 200, 0)  # BGR green
COLOR_FROZEN = (255, 160, 0)  # BGR orange
COLOR_FINETUNED = (255, 80, 80)  # BGR light red
COLOR_BOX_EXTRA = (160, 160, 160)  # BGR gray

LINE_THICKNESS = 3


def _clamp(v, lo, hi):
    return max(lo, min(hi, v))


def draw_box(image, box, color, label=None, thickness=LINE_THICKNESS):
    """Draw one rectangle (x1, y1, x2, y2 in pixels) on a copy of image."""
    out = image.copy()
    x1, y1, x2, y2 = (int(round(v)) for v in box)
    h, w = out.shape[:2]
    x1, y1 = _clamp(x1, 0, w), _clamp(y1, 0, h)
    x2, y2 = _clamp(x2, 0, w), _clamp(y2, 0, h)
    cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)
    if label:
        (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        ty = y1 - 6 if y1 - 6 - text_h >= 0 else y2 + text_h + 6
        ty = _clamp(ty, 0, h - text_h)
        cv2.rectangle(out, (x1, ty), (x1 + text_w + 8, ty + text_h + 8), color, -1)
        cv2.putText(out, label, (x1 + 4, ty + text_h + 2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return out


def draw_detections(image, detections, color, top_label=True, max_labels=3):
    """Overlay predicted boxes; label the top few with confidence."""
    out = image
    sorted_dets = sorted(detections, key=lambda d: d.confidence, reverse=True)
    for rank, det in enumerate(sorted_dets):
        label = f"{det.confidence:.2f}" if rank < max_labels else None
        if rank == 0 and top_label:
            label = f"top {det.confidence:.2f}"
        box_color = color if rank < max_labels else COLOR_BOX_EXTRA
        out = draw_box(out, (det.x1, det.y1, det.x2, det.y2), box_color, label)
    return out


def crop_box(image, box):
    """Cut a box from an image, clamped to bounds."""
    h, w = image.shape[:2]
    x1, y1, x2, y2 = (int(round(v)) for v in box)
    x1, y1 = _clamp(x1, 0, w), _clamp(y1, 0, h)
    x2, y2 = _clamp(x2, 0, w), _clamp(y2, 0, h)
    if x2 <= x1 or y2 <= y1:
        return None
    return image[y1:y2, x1:x2]


def gray_to_bgr(image):
    """Make a 3-channel BGR array for display (identity if already BGR)."""
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.ndim == 3 and image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    return image
