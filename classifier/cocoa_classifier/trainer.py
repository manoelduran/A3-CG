import json
from pathlib import Path

import numpy as np
from joblib import dump
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .data_loader import load_training_samples


def train(data_dir: Path, out_dir: Path, single_bean: bool = True):
    out_dir.mkdir(parents=True, exist_ok=True)
    features_matrix, class_labels, classes = load_training_samples(data_dir)

    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("svm", SVC(kernel="rbf", probability=True, C=10.0, gamma="scale")),
        ]
    )

    class_counts = np.unique(class_labels, return_counts=True)[1]
    n_splits = min(5, max(2, int(np.min(class_counts))))
    if n_splits > 1:
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        scores = cross_val_score(
            pipe, features_matrix, class_labels, cv=cv, scoring="accuracy"
        )
        print(f"CV accuracy: mean={scores.mean():.3f} ± {scores.std():.3f}")

    pipe.fit(features_matrix, class_labels)

    dump(pipe, out_dir / "model.pkl")
    with open(out_dir / "classes.json", "w") as f:
        json.dump(classes, f, indent=2)
