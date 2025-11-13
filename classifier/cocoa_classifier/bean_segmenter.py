import cv2
import numpy as np
from .segment_params import SegmentParams


def segment_beans(image: np.ndarray, params: SegmentParams) -> list[np.ndarray]:
    """
    Segments high-contrast seeds from a light background using Otsu thresholding.
    Returns a list of valid contours.
    """
    gray = _convert_to_grayscale(image)
    mask = _apply_otsu_threshold(gray)
    cleaned_mask = _clean_mask(mask)
    contours = _find_contours(cleaned_mask)
    valid_contours = _filter_contours_by_area(contours, params)

    _save_debug_images(gray, image, cleaned_mask, valid_contours)

    return valid_contours


def _convert_to_grayscale(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def _apply_otsu_threshold(gray: np.ndarray) -> np.ndarray:
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    return mask


def _clean_mask(mask: np.ndarray) -> np.ndarray:
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)


def _find_contours(mask: np.ndarray) -> list[np.ndarray]:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return list(contours)


def _filter_contours_by_area(
    contours: list[np.ndarray], params: SegmentParams
) -> list[np.ndarray]:
    return [
        c for c in contours if params.min_area <= cv2.contourArea(c) <= params.max_area
    ]


def _save_debug_images(
    gray: np.ndarray, image: np.ndarray, mask: np.ndarray, contours: list[np.ndarray]
) -> None:
    debug_image = image.copy()
    for c in contours:
        (x, y, w, h) = cv2.boundingRect(c)
        cv2.rectangle(debug_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imwrite("steps/0_grayscale.png", gray)
    cv2.imwrite("steps/1_simple_result.png", debug_image)
    cv2.imwrite("steps/2_otsu_mask.png", mask)
