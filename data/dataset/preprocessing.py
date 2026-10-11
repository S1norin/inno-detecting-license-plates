# CCPD OCR preprocessing: none, gray+contrast (CLAHE), and Canny.
import cv2
import numpy as np


def preprocess_plate(image, mode='none', rgb3=False):
    # Input: BGR uint8 HxWx3 or grayscale uint8. Output: uint8, BGR/grayscale by default; RGB 3ch if rgb3=True.
    if image is None or image.size == 0:
        raise ValueError('Empty plate image')
    if image.dtype != np.uint8:
        raise TypeError('Expected uint8 image')
    if image.ndim == 2:
        gray = image
        bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.ndim == 3 and image.shape[2] == 3:
        bgr = image
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    else:
        raise ValueError('Expected grayscale or BGR 3-channel image')
    if mode == 'none':
        out = bgr.copy() if image.ndim == 3 else gray.copy()
    elif mode == 'gray_contrast':
        out = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(gray)
    elif mode == 'canny':
        out = cv2.Canny(gray, threshold1=100, threshold2=200)
    else:
        raise ValueError('Unknown mode: '+str(mode))
    if rgb3:
        if out.ndim == 2:
            return cv2.cvtColor(out, cv2.COLOR_GRAY2RGB)
        return cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
    return out
