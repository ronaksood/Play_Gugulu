# =============================================================================
# models/engagement_classifier.py
# MODEL 2: Engagement / Fatigue / Attention Classifier
#
# Architecture: Single XGBClassifier, 3-class softmax:
#     Class 0 = fatigued
#     Class 1 = neutral
#     Class 2 = engaged
#
# Calibration: CalibratedClassifierCV(method="isotonic")
# Class balance: sample_weight computed from inverse class frequency
# Evaluation: macro F1, confusion matrix, calibration curve
# =============================================================================

import os
import joblib
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    ENGAGEMENT_CLASS_LABELS,
    ENGAGEMENT_LABEL_TO_IDX,
    FATIGUE_CONFIDENCE_SOFT_ALERT,
    FATIGUE_CONFIDENCE_HARD_STOP,
    MODEL_DIR,
    ENGAGEMENT_MODEL_PATH,
)
from data.feature_engineering import extract_engagement_features, get_engagement_feature_names


class EngagementClassifier:
    """
    Predicts the child's current engagement state from session-level features.

    Inference output
    ----------------
    {
        "engagement_state":  "engaged",          # "engaged" | "neutral" | "fatigued"
        "fatigue_risk":      0.08,               # P(fatigued)
        "attention_level":   0.87,               # P(engaged)
        "confidence":        0.91,               # max class probability
        "probabilities": {
            "engaged":   0.91,
            "neutral":   0.07,
            "fatigued":  0.02
        },
        "alert_level":       "none",             # "none" | "soft" | "hard"
        "cold_start":        false
    }

    alert_level is deterministic from fatigue_risk:
        fatigue_risk > FATIGUE_CONFIDENCE_HARD_STOP → "hard"  (end session)
        fatigue_risk > FATIGUE_CONFIDENCE_SOFT_ALERT → "soft" (shorten tasks)
        otherwise → "none"
    """

    def __init__(self, model_path: str = ENGAGEMENT_MODEL_PATH):
        self.model_path     = model_path
        self._model         = None
        self._feature_names = get_engagement_feature_names()
        self._load_model()

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def predict(
        self,
        events: list[dict],
        child_profile: dict,
        session_elapsed_seconds: float,
        session_allowed_seconds: float,
    ) -> dict:
        """
        Predict engagement state from current session events.

        Parameters
        ----------
        events                   : gameplay event list (current session)
        child_profile            : child profile dict
        session_elapsed_seconds  : seconds elapsed since session start
        session_allowed_seconds  : total allowed session duration in seconds
        """
        if not events:
            return self._cold_start_output()

        feats = extract_engagement_features(
            events,
            child_profile,
            session_elapsed_seconds,
            session_allowed_seconds,
        )

        if self._model is None:
            return self._rule_based_fallback(feats)

        X    = _feats_to_vector(feats, self._feature_names)

        try:
            proba = self._model.predict_proba([X])[0]   # shape [3]
            # Class order: 0=fatigued, 1=neutral, 2=engaged
            p_fatigued = float(proba[0])
            p_neutral  = float(proba[1])
            p_engaged  = float(proba[2])
            confidence = float(max(proba))
            pred_idx   = int(proba.argmax())
            pred_label = ENGAGEMENT_CLASS_LABELS[pred_idx]

            return {
                "engagement_state":  pred_label,
                "fatigue_risk":      round(p_fatigued, 4),
                "attention_level":   round(p_engaged,  4),
                "confidence":        round(confidence, 4),
                "probabilities": {
                    "fatigued": round(p_fatigued, 4),
                    "neutral":  round(p_neutral,  4),
                    "engaged":  round(p_engaged,  4),
                },
                "alert_level":  _compute_alert_level(p_fatigued),
                "cold_start":   False,
            }

        except Exception as e:
            return {**self._rule_based_fallback(feats), "_fallback_reason": str(e)}

    def predict_batch(
        self,
        batch: list[tuple[list, dict, float, float]],
    ) -> list[dict]:
        """Batch: each element is (events, child_profile, elapsed, allowed)."""
        return [self.predict(*args) for args in batch]

    def is_trained(self) -> bool:
        return self._model is not None

    # -----------------------------------------------------------------------
    # Internal
    # -----------------------------------------------------------------------

    def _load_model(self):
        if os.path.exists(self.model_path):
            try:
                self._model = joblib.load(self.model_path)
            except Exception:
                self._model = None

    def _cold_start_output(self) -> dict:
        return {
            "engagement_state":  "engaged",
            "fatigue_risk":      0.10,
            "attention_level":   0.75,
            "confidence":        0.40,
            "probabilities":     {"fatigued": 0.10, "neutral": 0.25, "engaged": 0.65},
            "alert_level":       "none",
            "cold_start":        True,
        }

    def _rule_based_fallback(self, feats: dict) -> dict:
        """
        Deterministic heuristic for when model is not trained yet.
        This is the cold-start / no-model path — still useful for demo.
        Implements the same decision logic that trained data was weakly labelled with.
        """
        rt_trend  = feats.get("response_time_trend", 0)
        err_rate  = feats.get("error_rate_last_5", 0.2)
        pause_max = feats.get("pause_duration_max", 0)
        elapsed   = feats.get("session_elapsed_seconds", 0)
        dur_ratio = feats.get("session_duration_ratio", 0)
        hint_r    = feats.get("hint_usage_rate_session", 0.2)

        # Heuristic fatigue score (0–1)
        fatigue_score = 0.0
        fatigue_score += min(rt_trend / 100.0, 0.30)         # slowing RT
        fatigue_score += min(err_rate * 0.5, 0.25)           # high errors
        fatigue_score += min(pause_max / 20000.0, 0.20)      # long pauses
        fatigue_score += min(dur_ratio * 0.25, 0.15)         # session length
        fatigue_score += min(hint_r * 0.2, 0.10)             # hint dependency
        fatigue_score = max(0.0, min(1.0, fatigue_score))

        engaged_score  = max(0.0, 1.0 - fatigue_score * 1.4)
        neutral_score  = max(0.0, 1.0 - abs(fatigue_score - 0.5) * 2)
        total = engaged_score + neutral_score + fatigue_score + 1e-9

        p_eng = engaged_score / total
        p_neu = neutral_score / total
        p_fat = fatigue_score / total

        if p_fat > max(p_eng, p_neu):
            state = "fatigued"
        elif p_neu > p_eng:
            state = "neutral"
        else:
            state = "engaged"

        return {
            "engagement_state":  state,
            "fatigue_risk":      round(p_fat, 4),
            "attention_level":   round(p_eng, 4),
            "confidence":        round(max(p_fat, p_neu, p_eng), 4),
            "probabilities": {
                "fatigued": round(p_fat, 4),
                "neutral":  round(p_neu, 4),
                "engaged":  round(p_eng, 4),
            },
            "alert_level":   _compute_alert_level(p_fat),
            "cold_start":    True,
        }


# ---------------------------------------------------------------------------
# Model Spec (used by training script)
# ---------------------------------------------------------------------------

def build_xgb_engagement_model():
    """
    Returns a calibrated XGBClassifier for 3-class engagement prediction.

    Spec
    ----
    Base learner     : XGBClassifier
    Objective        : multi:softprob
    num_class        : 3  (fatigued=0, neutral=1, engaged=2)
    n_estimators     : 200
    max_depth        : 4
    learning_rate    : 0.05
    subsample        : 0.8
    colsample_bytree : 0.8
    min_child_weight : 3
    Class imbalance  : sample_weight = 1 / class_freq (passed to .fit())
    Calibration      : CalibratedClassifierCV(method="isotonic", cv=3)
    Evaluation       : macro F1, confusion matrix, per-class calibration curve
    """
    try:
        from xgboost import XGBClassifier
        from sklearn.calibration import CalibratedClassifierCV
    except ImportError:
        raise ImportError("xgboost and scikit-learn required: pip install xgboost scikit-learn")

    base = XGBClassifier(
        objective         = "multi:softprob",
        num_class         = 3,
        n_estimators      = 200,
        max_depth         = 4,
        learning_rate     = 0.05,
        subsample         = 0.8,
        colsample_bytree  = 0.8,
        min_child_weight  = 3,
        gamma             = 0.1,
        reg_alpha         = 0.05,
        reg_lambda        = 1.0,
        random_state      = 42,
        eval_metric       = "mlogloss",
        verbosity         = 0,
        use_label_encoder = False,
    )

    calibrated = CalibratedClassifierCV(
        estimator = base,
        method    = "isotonic",
        cv        = 3,
    )
    return calibrated


def compute_sample_weights(labels) -> list:
    """
    Compute inverse-frequency sample weights for class imbalance handling.
    Pass the result to XGBClassifier.fit(X, y, sample_weight=weights).
    """
    from collections import Counter
    counts = Counter(labels)
    total  = len(labels)
    return [total / (len(counts) * counts[y]) for y in labels]


# ---------------------------------------------------------------------------
# Private utilities
# ---------------------------------------------------------------------------

def _feats_to_vector(feats: dict, feature_names: list[str]) -> list[float]:
    return [float(feats.get(name, 0.0)) for name in feature_names]


def _compute_alert_level(fatigue_risk: float) -> str:
    if fatigue_risk >= FATIGUE_CONFIDENCE_HARD_STOP:
        return "hard"
    elif fatigue_risk >= FATIGUE_CONFIDENCE_SOFT_ALERT:
        return "soft"
    return "none"
