# =============================================================================
# training/train_skill_mastery.py
# Train 11 XGBClassifiers (one per skill) on synthetic data.
# Saves calibrated models to models/artifacts/{skill}_mastery.joblib
#
# Run: python training/train_skill_mastery.py
# =============================================================================

import os
import sys
import json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import SKILLS, MODEL_DIR, skill_model_path
from data.training_schema import SyntheticDataGenerator
from data.feature_engineering import get_skill_feature_names
from models.skill_mastery import build_xgb_skill_model


def train_and_save_all(
    n_samples: int = 8000,
    output_dir: str = MODEL_DIR,
    verbose: bool = True,
) -> dict:
    """
    Train one calibrated XGBClassifier per skill and save to disk.

    Training strategy
    -----------------
    - Temporal split is approximated by sessions_completed feature:
      80% of samples (lower session counts) → train
      20% (higher session counts) → validation
    - Class imbalance: scale_pos_weight in XGBoost + class-weight-aware split
    - Calibration: CalibratedClassifierCV(isotonic, cv=3) applied on training fold
    - Evaluation: AUC-ROC, Brier score (calibration), classification report

    Returns
    -------
    dict: {skill: {"auc": float, "brier": float, "n_train": int, "n_val": int}}
    """
    try:
        import pandas as pd
        import numpy as np
        import joblib
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import roc_auc_score, brier_score_loss, classification_report
    except ImportError:
        raise ImportError(
            "Required: pip install xgboost scikit-learn pandas numpy joblib"
        )

    os.makedirs(output_dir, exist_ok=True)
    feature_names = get_skill_feature_names()

    if verbose:
        print(f"Generating {n_samples} synthetic samples...")

    gen = SyntheticDataGenerator(seed=42)
    X_all, y_all = gen.generate_skill_mastery_dataset(n_samples=n_samples)

    results = {}

    for skill in SKILLS:
        if verbose:
            print(f"\n{'='*50}")
            print(f"Training: {skill.upper()}")

        # Filter to this skill's rows
        mask    = y_all["skill"] == skill
        X_skill = X_all[mask].copy()
        y_skill = y_all[mask]["mastery_label"].copy()

        if len(X_skill) < 50:
            if verbose:
                print(f"  Skipping {skill} — insufficient samples ({len(X_skill)})")
            continue

        # Align feature columns to expected order
        for col in feature_names:
            if col not in X_skill.columns:
                X_skill[col] = 0.0
        X_skill = X_skill[feature_names]

        # Temporal split: use sessions_completed as a proxy for time ordering
        X_sorted = X_skill.sort_values("sessions_completed")
        y_sorted = y_skill.loc[X_sorted.index]
        split_idx = int(len(X_sorted) * 0.80)
        X_train, X_val = X_sorted.iloc[:split_idx], X_sorted.iloc[split_idx:]
        y_train, y_val = y_sorted.iloc[:split_idx], y_sorted.iloc[split_idx:]

        if verbose:
            n_pos = int(y_train.sum())
            n_neg = len(y_train) - n_pos
            print(f"  Train: {len(X_train)} samples  (pos={n_pos}, neg={n_neg})")
            print(f"  Val:   {len(X_val)} samples")

        # Handle class imbalance via scale_pos_weight
        n_neg = (y_train == 0).sum()
        n_pos = (y_train == 1).sum()
        spw   = n_neg / max(n_pos, 1)

        # Build model with correct scale_pos_weight
        from xgboost import XGBClassifier
        from sklearn.calibration import CalibratedClassifierCV

        base = XGBClassifier(
            objective          = "binary:logistic",
            n_estimators       = 150,
            max_depth          = 4,
            learning_rate      = 0.05,
            subsample          = 0.8,
            colsample_bytree   = 0.8,
            min_child_weight   = 5,
            gamma              = 0.1,
            reg_alpha          = 0.1,
            reg_lambda         = 1.0,
            scale_pos_weight   = spw,
            random_state       = 42,
            eval_metric        = "auc",
            verbosity          = 0,
            use_label_encoder  = False,
        )

        model = CalibratedClassifierCV(estimator=base, method="isotonic", cv=3)

        model.fit(X_train.values, y_train.values)

        # Evaluate
        y_prob = model.predict_proba(X_val.values)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)

        auc    = roc_auc_score(y_val, y_prob) if len(y_val.unique()) > 1 else 0.0
        brier  = brier_score_loss(y_val, y_prob)

        if verbose:
            print(f"  AUC-ROC : {auc:.4f}")
            print(f"  Brier   : {brier:.4f}  (lower is better, 0=perfect)")
            unique_labels = sorted(y_val.unique())
            target_names  = ["not_mastered", "mastered"][:len(unique_labels)]
            print(classification_report(y_val, y_pred, labels=unique_labels, target_names=target_names, zero_division=0))

        # Save model
        path = skill_model_path(skill)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(model, path)

        if verbose:
            print(f"  Saved -> {path}")

        results[skill] = {
            "auc":    round(auc, 4),
            "brier":  round(brier, 4),
            "n_train": int(len(X_train)),
            "n_val":   int(len(X_val)),
        }

    # Save results summary
    summary_path = os.path.join(output_dir, "skill_mastery_training_results.json")
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)

    if verbose:
        print(f"\n{'='*50}")
        print("TRAINING COMPLETE")
        print(f"Results saved to {summary_path}")
        mean_auc = sum(r["auc"] for r in results.values()) / max(len(results), 1)
        print(f"Mean AUC across skills: {mean_auc:.4f}")

    return results


if __name__ == "__main__":
    train_and_save_all(n_samples=8000, verbose=True)
