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
    
    # Collect all features first for analysis
    all_features = []
    for cnt in contours:
        features = contour_features(image, cnt)
        all_features.append(features)
    
    if all_features:
        # Check feature variance to see if beans are actually similar
        features_array = np.array(all_features)
        feature_std = np.std(features_array, axis=0)
        print(f"Feature std across beans: min={feature_std.min():.2f}, max={feature_std.max():.2f}, mean={feature_std.mean():.2f}")
    
    for i, (cnt, features) in enumerate(zip(contours, all_features)):
        # Use decision function scores instead of Platt-scaled probabilities
        # This preserves relative differences better and is more interpretable
        try:
            svm_step = model.named_steps.get('svm')
            if svm_step is not None and hasattr(svm_step, 'decision_function'):
                scaler = model.named_steps.get('scaler')
                if scaler is not None:
                    scaled_features = scaler.transform([features])
                    decision_score = svm_step.decision_function(scaled_features)[0]
                else:
                    decision_score = svm_step.decision_function([features])[0]
                
                # Convert decision score to probability using sigmoid with temperature scaling
                # Temperature helps spread out similar scores for better differentiation
                # Lower temperature = more spread, higher = more compressed
                # Using 0.3 to better differentiate similar decision scores
                temperature = 0.3  # Lower values spread probabilities more for similar scores
                scaled_score = decision_score / temperature
                prob_class1 = 1 / (1 + np.exp(-scaled_score))
                
                # Determine which class based on decision score
                # Negative = class 0, positive = class 1
                if decision_score < 0:
                    yhat = 0
                    confidence = 1 - prob_class1  # Probability of class 0
                else:
                    yhat = 1
                    confidence = prob_class1  # Probability of class 1
                
                # Create probability dict for consistency
                probability = np.array([1 - prob_class1, prob_class1])
                
            else:
                # Fallback to standard predict_proba if decision_function not available
                probability = model.predict_proba([features])[0]
                yhat = int(np.argmax(probability))
                confidence = float(probability[yhat])
                decision_score = None
        except Exception:
            # Fallback to standard predict_proba
            probability = model.predict_proba([features])[0]
            yhat = int(np.argmax(probability))
            confidence = float(probability[yhat])
            decision_score = None
        
        label = classes[yhat]
        
        # Log for debugging
        prob_dict = {classes[j]: float(prob) for j, prob in enumerate(probability)}
        if decision_score is not None:
            print(f"Bean {i}: {label} (conf={confidence:.4f}), "
                  f"probs={prob_dict}, decision_score={decision_score:.4f}")
        else:
            print(f"Bean {i}: {label} (conf={confidence:.4f}), probs={prob_dict}")

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
