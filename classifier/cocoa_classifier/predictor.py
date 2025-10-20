from typing import Any, TypedDict

import cv2
import numpy as np
from cv2.typing import MatLike

from .bean_segmenter import get_contours
from .feature_contourer import contour_features


class PredictionResultRow(TypedDict):
    idx: int
    x: int
    y: int
    w: int
    h: int
    pred_class: str
    confidence: float


def predict(
    file: bytes,
    model: Any,
    classes: list[str],
    single_bean: bool = False,
):
    image = _decode_image(file)
    contours = get_contours(image, single_bean)

    results: list[PredictionResultRow] = []
    overlay = image.copy()
    for i, cnt in enumerate(contours):
        features = contour_features(image, cnt)
        probability = model.predict_proba([features])[0]
        yhat = int(np.argmax(probability))
        label = classes[yhat]
        confidence = float(probability[yhat])

        x, y, width, height = cv2.boundingRect(cnt)
        cv2.rectangle(
            overlay,
            (x, y),
            (x + width, y + height),
            (0, 255, 0),
            2,
        )
        text = f"{label} {confidence:.2f}"
        cv2.putText(
            overlay,
            text,
            (x, y + 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 0, 255),
            2,
            cv2.LINE_AA,
        )

        results.append(
            {
                "idx": i,
                "x": x,
                "y": y,
                "w": width,
                "h": height,
                "pred_class": label,
                "confidence": confidence,
            }
        )

    return overlay, results


def _decode_image(file: bytes) -> MatLike:
    return cv2.imdecode(
        np.frombuffer(file, dtype=np.uint8),
        cv2.IMREAD_COLOR,
    )
