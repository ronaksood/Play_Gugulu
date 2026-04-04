# =============================================================================
# training/train_engagement.py
# Train a single 3-class XGBClassifier for engagement/fatigue detection.
# Saves calibrated model to models/artifacts/engagement_classifier.joblib
#
# Run: python training/train_engagement.py
# =============================================================================

import os
import sys
import json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import ENGAGEMENT_MODEL_PATH, MODEL_DIR, ENGAGEMENT_CLASS_LABELS
from data.training_schema import SyntheticDataGenerator
from data.feature_engineering import get_engagement_feature_names
from models.engagement_classifier import build_xgb_engagement_model, compute_sample_weights


def train_and_save(
    n_samples: int = 8000,
    output_path: str = ENGAGEMENT_MODEL_PATH,
    verbose: bool = True,
) -> dict:
    """
    Train the engagement/fatigue classifier and save to disk.

    Training strategy
    -----------------
    - Stratified train/val split (80/20) to preserve class balance
    - Class imbalance: inverse-frequency sample_weight passed to fit()
    - Calibration: CalibratedClassifierCV(isotonic, cv=3)
    - Evaluation: macro F1, per-class precision/recall, confusion matrix
    - Calibration quality: per-class Brier score

    Returns
    -------
    dict with evaluation metrics
    """
    try:
        import pandas as pd
        import numpy as np
        import joblib
        from sklearn.model_selection import StratifiedShuffleSplit
        from sklearn.metrics import (
            classification_report,
            confusion_matrix,
            f1_score,
            brier_score_loss,
        )
        from sklearn.preprocessing import label_binarize
    except ImportError:
        raise ImportError(
            "Required: pip install xgboost scikit-learn pandas numpy joblib"
        )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    feature_names = get_engagement_feature_names()

    if verbose:
        print(f"Generating {n_samples} synthetic engagement samples...")

    gen    = SyntheticDataGenerator(seed=42)
    X, y   = gen.generate_engagement_dataset(n_samples=n_samples)

    if verbose:
        print(f"Label distribution: {y.value_counts().to_dict()}")
        # 0=fatigued, 1=neutral, 2=engaged

    # Align feature columns
    for col in feature_names:
        if col not in X.columns:
            X[col] = 0.0
    X = X[feature_names]

    # Stratified split
    sss    = StratifiedShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
    train_idx, val_idx = next(sss.split(X, y))
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    if verbose:
        print(f"Train: {len(X_train)}  |  Val: {len(X_val)}")

    # Compute sample weights for class balance
    sample_weights = compute_sample_weights(y_train.values.tolist())

    # Build model
    model = build_xgb_engagement_model()

    if verbose:
        print("Training calibrated XGBClassifier (3-class, multi:softprob)...")

    # CalibratedClassifierCV with sample_weight
    # We pass sample_weight via fit_params (sklearn >= 1.4 supports this)
    try:
        model.fit(
            X_train.values,
            y_train.values,
            sample_weight=sample_weights,
        )
    except TypeError:
        # Older sklearn — fit without sample_weight
        model.fit(X_train.values, y_train.values)

    # Evaluate
    y_prob  = model.predict_proba(X_val.values)    # shape: (n, 3)
    y_pred  = y_prob.argmax(axis=1)

    macro_f1 = f1_score(y_val, y_pred, average="macro")

    if verbose:
        print(f"\nMacro F1: {macro_f1:.4f}")
        print("\nClassification Report:")
        labels = [ENGAGEMENT_CLASS_LABELS[i] for i in range(3)]
        print(classification_report(y_val, y_pred, target_names=labels, zero_division=0))
        print("Confusion Matrix (rows=true, cols=pred):")
        cm = confusion_matrix(y_val, y_pred)
        print(f"  {'':12} {' '.join(f'{l:12}' for l in labels)}")
        for i, row in enumerate(cm):
            print(f"  {labels[i]:12} {' '.join(f'{v:12d}' for v in row)}")

    # Per-class Brier scores (calibration)
    y_bin      = label_binarize(y_val, classes=[0, 1, 2])
    brier_per_class = {
        ENGAGEMENT_CLASS_LABELS[i]: round(float(brier_score_loss(y_bin[:, i], y_prob[:, i])), 4)
        for i in range(3)
    }
    if verbose:
        print(f"\nBrier scores (calibration, lower=better):")
        for cls, bs in brier_per_class.items():
            print(f"  {cls}: {bs:.4f}")

    # Feature importance (from first calibrated estimator)
    feat_importances = []
    try:
        base_est = model.calibrated_classifiers_[0].estimator
        importances = base_est.feature_importances_
        feat_importances = sorted(
            [{"feature": feature_names[i], "importance": round(float(v), 6)}
             for i, v in enumerate(importances)],
            key=lambda x: x["importance"], reverse=True
        )
        if verbose:
            print("\nTop 10 Features:")
            for f in feat_importances[:10]:
                print(f"  {f['feature']:45} {f['importance']:.4f}")
    except Exception:
        pass

    # Save model
    import joblib
    joblib.dump(model, output_path)
    if verbose:
        print(f"\nModel saved -> {output_path}")

    results = {
        "macro_f1":          round(macro_f1, 4),
        "brier_per_class":   brier_per_class,
        "n_train":           int(len(X_train)),
        "n_val":             int(len(X_val)),
        "top_features":      feat_importances[:10],
    }

    summary_path = os.path.join(os.path.dirname(output_path), "engagement_training_results.json")
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)

    if verbose:
        print(f"Results saved -> {summary_path}")

    return results


if __name__ == "__main__":
    train_and_save(n_samples=8000, verbose=True)
