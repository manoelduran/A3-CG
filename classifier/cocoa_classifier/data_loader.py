from pathlib import Path
import cv2
import numpy as np
from .bean_segmenter import segment_beans
from .segment_params import SegmentParams
from .feature_contourer import contour_features
from .helpers import get_blurred_gray


def load_training_samples(
    data_dir: Path,
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

            contours = segment_beans(
                image, SegmentParams(min_area=1000, max_area=100000)
            )

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
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
    )
    return threshold


def _find_contours(image: np.ndarray) -> tuple:
    contours, hierarchy = cv2.findContours(
        image,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    return contours, hierarchy
