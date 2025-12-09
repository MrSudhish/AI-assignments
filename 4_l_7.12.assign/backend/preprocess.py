# backend/preprocess.py
import cv2
import numpy as np

IMG_SIZE = (128, 128)

def load_and_preprocess(img_or_path):
    """
    Accepts either:
      - a numpy image (as returned by cv2.imdecode), or
      - a filesystem path string (to read via cv2.imread)
    Returns: 1D float32 feature vector (normalized)
    """
    # If it's a path (string), load it
    if isinstance(img_or_path, str):
        img = cv2.imread(img_or_path)
        if img is None:
            raise ValueError(f"Could not read image from path: {img_or_path}")
    else:
        # assume already a numpy array (BGR)
        img = img_or_path

    # If image is BGR (cv2 default), convert to RGB (not strictly necessary if consistent)
    try:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    except Exception:
        # if conversion fails, assume image already appropriate
        pass

    img = cv2.resize(img, IMG_SIZE)
    img = img.astype("float32") / 255.0
    return img.flatten()
