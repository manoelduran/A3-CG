from pathlib import Path

import cv2
import numpy as np
from cv2.typing import MatLike

from .bean_segmenter import get_contours
from .feature_contourer import contour_features
from .helpers import get_blurred_gray


def load_training_samples(
    data_dir: Path,
    single_bean: bool = True,
) -> tuple[
    np.ndarray,
    np.ndarray,
    list[str],
]:
    feature_vectors: list[np.ndarray] = []
    class_labels: list[int] = []

    classes = sorted([d.name for d in data_dir.iterdir() if d.is_dir()])
    if not classes:
        raise RuntimeError(f"No class folders found in {data_dir}")

    for idx, cls in enumerate(classes):
        image_paths = sorted((data_dir / cls).glob("*.*"))
        for path in image_paths:
            encoded_image = _read_file(path)
            image = _decode_image(encoded_image)
            if image is None:
                continue

            contours = get_contours(image, single_bean)

            if not contours:
                threshold = _find_threshold(image)
                contours = _find_contours(threshold)
                contours = [c for c in contours if 1000 <= cv2.contourArea(c) <= 100000]

            if contours:
                contour = max(contours, key=cv2.contourArea)
                features = contour_features(image, contour)
                feature_vectors.append(features)
                class_labels.append(idx)

    if not feature_vectors:
        raise RuntimeError("No training samples extracted. Check images.")
    return np.vstack(feature_vectors), np.array(class_labels), classes


def _read_file(path: Path) -> np.ndarray:
    return np.fromfile(str(path), dtype=np.uint8)


def _decode_image(arr: np.ndarray) -> np.ndarray:
    return cv2.imdecode(
        arr,
        cv2.IMREAD_COLOR,
    )


def _find_threshold(image: np.ndarray) -> np.ndarray:
    blur = get_blurred_gray(image)
    _, threshold = cv2.threshold(
        blur,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )
    return threshold


def _find_contours(image: np.ndarray) -> MatLike:
    _, contours = cv2.findContours(
        image,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    return contours
