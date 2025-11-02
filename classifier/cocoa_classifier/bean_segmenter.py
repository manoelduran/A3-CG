import cv2
import numpy as np
from cv2.typing import MatLike

from .helpers import convert_to_bgr, convert_to_lab, get_blurred_gray
from .segment_params import SegmentParams


def segment_beans(image: np.ndarray, params: SegmentParams) -> list[np.ndarray]:
    """Return binary mask and list of contours for each bean after watershed splitting."""
    blur = _preprocess_to_gray(image)
    white_foreground = _binarize_to_foreground(blur)
    opened = _open_foreground(white_foreground, params.open_ksize)
    sure_background = _dilate_for_sure_background(opened, params.sure_bg_dilate)
    sure_foreground = _distance_core_sure_foreground(opened, params.distance_thresh)
    unknown = _compute_unknown(sure_background, sure_foreground)
    markers = _compute_markers(sure_foreground, unknown)

    _apply_watershed_in_place(image, markers)

    contours: list[np.ndarray] = _extract_valid_contours(
        markers,
        params.min_area,
        params.max_area,
    )

    return contours


def segment_single_bean(image: np.ndarray, params: SegmentParams) -> list[np.ndarray]:
    """Segment single bean from training image using simple thresholding."""
    blur = _preprocess_to_gray(image)
    threshold = _binarize_to_foreground(blur)
    threshold = _open_foreground(threshold, params.open_ksize)

    # Find contours
    contours, _ = cv2.findContours(
        threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    # Filter by area and return the largest valid contour
    valid_contours = [
        c for c in contours if params.min_area <= cv2.contourArea(c) <= params.max_area
    ]

    return valid_contours


def _normalize_contrast(img_bgr: np.ndarray) -> np.ndarray:
    """Apply CLAHE to L channel in Lab for gentle contrast normalization."""
    lab = convert_to_lab(img_bgr)
    brightness, green_red, blue_yellow = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_brightness = clahe.apply(brightness)
    lab_enhanced = cv2.merge([enhanced_brightness, green_red, blue_yellow])
    return convert_to_bgr(lab_enhanced)


def _preprocess_to_gray(image: np.ndarray) -> np.ndarray:
    image = _normalize_contrast(image)
    return get_blurred_gray(image)


def _binarize_to_foreground(image: np.ndarray) -> np.ndarray:
    _, threshold = cv2.threshold(
        image,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )

    return threshold if threshold.mean() < 127 else cv2.bitwise_not(threshold)


def _open_foreground(
    image: np.ndarray,
    ksize: int,
) -> np.ndarray:
    k_shape = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    return cv2.morphologyEx(image, cv2.MORPH_OPEN, k_shape, iterations=1)


def _dilate_for_sure_background(
    image: np.ndarray,
    ksize: int,
) -> np.ndarray:
    kernel_shape = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    return cv2.dilate(image, kernel_shape, iterations=2)


def _distance_core_sure_foreground(
    image: np.ndarray,
    distance_thresh: float,
) -> np.ndarray:
    distance = cv2.distanceTransform(image, cv2.DIST_L2, 5)
    max_distance = distance.max()
    normal_distance = distance / max_distance if max_distance > 0 else distance
    return (normal_distance > distance_thresh).astype(np.uint8) * 255


def _compute_unknown(
    sure_background: np.ndarray,
    sure_foreground: np.ndarray,
) -> np.ndarray:
    """Resolves unknown regions between sure background and sure foreground to form boundaries."""
    return cv2.subtract(
        sure_background,
        sure_foreground,
    )


def _compute_markers(
    sure_foreground: np.ndarray,
    unknown: np.ndarray,
) -> np.ndarray:
    _, markers = cv2.connectedComponents(sure_foreground)
    markers = markers + 1
    markers[unknown == 255] = 0
    return markers


def _apply_watershed_in_place(image: np.ndarray, markers: np.ndarray):
    cv2.watershed(image, markers)


def get_contours(image: MatLike, single_bean: bool) -> list[np.ndarray]:
    if single_bean:
        return segment_single_bean(
            image,
            SegmentParams(
                min_area=1000,
                max_area=100000,
                open_ksize=5,
            ),
        )
    return segment_beans(
        image,
        SegmentParams(
            min_area=800,
            max_area=25_000,
            open_ksize=5,
            sure_bg_dilate=5,
            distance_thresh=0.05,
        ),
    )


def _extract_valid_contours(
    markers: np.ndarray, min_area: float, max_area: float
) -> list[np.ndarray]:
    contours: list[np.ndarray] = []
    for label in range(2, markers.max() + 1):
        comp = (markers == label).astype(np.uint8) * 255
        found, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in found:
            area = cv2.contourArea(c)
            if min_area <= area <= max_area:
                contours.append(c)
    return contours
