import math

import cv2
import numpy as np

from .helpers import convert_to_lab


def contour_features(image: np.ndarray, contour: np.ndarray) -> np.ndarray:
    """Extracts shape and color features for a single contour."""
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.drawContours(
        mask,
        [contour],
        -1,
        (255, 255, 255),
        -1,
    )

    area = cv2.contourArea(contour)
    perim = cv2.arcLength(contour, True)
    x, y, w, h = cv2.boundingRect(contour)
    aspect = w / (h + 1e-6)
    circularity = (4 * math.pi * area) / ((perim + 1e-6) ** 2)
    hull = cv2.convexHull(contour)
    solidity = area / (cv2.contourArea(hull) + 1e-6)

    eccentricity = 0.0
    if len(contour) >= 5:
        _, (major_axis, minor_axis), _ = cv2.fitEllipse(contour)
        a = max(major_axis, minor_axis) / 2.0
        b = min(major_axis, minor_axis) / 2.0
        if a > 1e-6:
            eccentricity = math.sqrt(1.0 - (b**2) / (a**2))

    lab = convert_to_lab(image)

    def masked_stats(ch):
        vals = ch[mask == 255]
        return (float(vals.mean()), float(vals.std())) if len(vals) > 0 else (0.0, 0.0)

    Lm, Ls = masked_stats(lab[:, :, 0])
    Am, As = masked_stats(lab[:, :, 1])
    Bm, Bs = masked_stats(lab[:, :, 2])

    features = [
        area,
        perim,
        aspect,
        circularity,
        solidity,
        eccentricity,
        Lm,
        Ls,
        Am,
        As,
        Bm,
        Bs,
    ]
    return np.array(features, dtype=np.float32)
