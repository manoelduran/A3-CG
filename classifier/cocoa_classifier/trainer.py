import json
from pathlib import Path

import numpy as np
from joblib import dump
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .data_loader import load_training_samples


def train(data_dir: Path, out_dir: Path, single_bean: bool = True):
    out_dir.mkdir(parents=True, exist_ok=True)
    features_matrix, class_labels, classes = load_training_samples(data_dir)

    # Report dataset statistics
    print(f"\n{'='*60}")
    print("Dataset Statistics")
    print(f"{'='*60}")
    print(f"Total samples: {len(class_labels)}")
    print(f"Features: {features_matrix.shape[1]}")
    
    # Class balance
    unique, counts = np.unique(class_labels, return_counts=True)
    print(f"\nClass distribution:")
    for cls_idx, count in zip(unique, counts):
        print(f"  {classes[cls_idx]}: {count} ({count/len(class_labels)*100:.1f}%)")
    
    # Feature statistics
    print(f"\nFeature statistics:")
    print(f"  Mean std across features: {np.std(features_matrix, axis=0).mean():.2f}")
    print(f"  Features with zero variance: {(np.std(features_matrix, axis=0) < 1e-6).sum()}")

    # Split data for evaluation
    X_train, X_test, y_train, y_test = train_test_split(
        features_matrix, class_labels, test_size=0.2, random_state=42, stratify=class_labels
    )
    print(f"\nTrain set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")

    # Hyperparameter tuning with GridSearchCV
    print(f"\n{'='*60}")
    print("Hyperparameter Tuning")
    print(f"{'='*60}")
    
    param_grid = {
        'svm__C': [0.1, 1.0, 10.0, 100.0],
        'svm__gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1.0],
    }
    
    base_pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("svm", SVC(kernel="rbf", probability=True)),
        ]
    )
    
    class_counts = np.unique(y_train, return_counts=True)[1]
    n_splits = min(5, max(2, int(np.min(class_counts))))
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    grid_search = GridSearchCV(
        base_pipe,
        param_grid,
        cv=cv,
        scoring='f1_weighted',
        n_jobs=-1,
        verbose=1,
    )
    
    print("Searching for best hyperparameters...")
    grid_search.fit(X_train, y_train)
    
    print(f"\nBest parameters: {grid_search.best_params_}")
    print(f"Best CV score (F1-weighted): {grid_search.best_score_:.4f}")
    
    # Use best model
    pipe = grid_search.best_estimator_

    # Cross-validation evaluation
    print(f"\n{'='*60}")
    print("Cross-Validation Results")
    print(f"{'='*60}")
    
    cv_scores_acc = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="accuracy")
    cv_scores_f1 = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="f1_weighted")
    cv_scores_prec = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="precision_weighted")
    cv_scores_rec = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="recall_weighted")
    
    print(f"Accuracy:  {cv_scores_acc.mean():.4f} ± {cv_scores_acc.std():.4f}")
    print(f"F1-score:  {cv_scores_f1.mean():.4f} ± {cv_scores_f1.std():.4f}")
    print(f"Precision: {cv_scores_prec.mean():.4f} ± {cv_scores_prec.std():.4f}")
    print(f"Recall:    {cv_scores_rec.mean():.4f} ± {cv_scores_rec.std():.4f}")

    # Train on full training set
    pipe.fit(X_train, y_train)
    
    # Test set evaluation
    print(f"\n{'='*60}")
    print("Test Set Evaluation")
    print(f"{'='*60}")
    
    y_pred = pipe.predict(X_test)
    
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=classes))
    
    # Feature importance (using SVM coefficients for linear kernel, or permutation importance)
    # For RBF kernel, we can't get direct feature importance, but we can check feature ranges
    print(f"\n{'='*60}")
    print("Feature Analysis")
    print(f"{'='*60}")
    
    feature_names = [
        'area', 'perimeter', 'aspect_ratio', 'circularity', 'solidity', 'eccentricity',
        'L_mean', 'L_std', 'A_mean', 'A_std', 'B_mean', 'B_std'
    ]
    
    # Check feature ranges by class
    for cls_idx, cls_name in enumerate(classes):
        cls_mask = class_labels == cls_idx
        cls_features = features_matrix[cls_mask]
        print(f"\n{cls_name} class feature ranges:")
        for i, feat_name in enumerate(feature_names):
            feat_vals = cls_features[:, i]
            print(f"  {feat_name:15s}: mean={feat_vals.mean():8.2f}, std={feat_vals.std():8.2f}, "
                  f"range=[{feat_vals.min():8.2f}, {feat_vals.max():8.2f}]")

    # Save model
    dump(pipe, out_dir / "model.pkl")
    with open(out_dir / "classes.json", "w") as f:
        json.dump(classes, f, indent=2)
    
    # Save training report
    report = {
        "best_params": grid_search.best_params_,
        "best_cv_score": float(grid_search.best_score_),
        "cv_accuracy": {
            "mean": float(cv_scores_acc.mean()),
            "std": float(cv_scores_acc.std()),
        },
        "cv_f1": {
            "mean": float(cv_scores_f1.mean()),
            "std": float(cv_scores_f1.std()),
        },
        "test_accuracy": float((y_pred == y_test).mean()),
        "test_f1": float(f1_score(y_test, y_pred, average='weighted')),
        "test_precision": float(precision_score(y_test, y_pred, average='weighted')),
        "test_recall": float(recall_score(y_test, y_pred, average='weighted')),
        "confusion_matrix": cm.tolist(),
    }
    
    with open(out_dir / "training_report.json", "w") as f:
        json.dump(report, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"Model saved to {out_dir}")
    print(f"Training report saved to {out_dir / 'training_report.json'}")
    print(f"{'='*60}\n")
