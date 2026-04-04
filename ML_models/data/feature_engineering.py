# =============================================================================
# data/feature_engineering.py
# Converts raw gameplay events + child profile into flat feature vectors
# for both XGBoost models.
# All outputs are plain Python dicts — no pandas dependency at inference time.
# =============================================================================

import math
from typing import Optional
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    SKILLS,
    RESPONSE_TIME_WINDOW,
    ACCURACY_WINDOW_SHORT,
    ACCURACY_WINDOW_LONG,
    MODALITY_ENCODING,
    DEFAULT_MODALITY,
    AGE_BANDS,
)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_skill_features(
    events: list[dict],
    skill: str,
    child_profile: dict,
) -> dict:
    """
    Build the feature vector for the Skill Mastery Classifier for one skill.

    Parameters
    ----------
    events : list of gameplay event dicts (chronological)
    skill  : the skill to compute features for
    child_profile : current profile dict (mastery_probs, preferred_modality, etc.)

    Returns
    -------
    dict — flat feature vector with stable string keys
    """
    skill_events = [e for e in events if e.get("skill") == skill]
    all_recent   = events[-SESSION_WINDOW_SIZE:] if len(events) > SESSION_WINDOW_SIZE else events

    feats = {}

    # --- Accuracy features ---
    feats["accuracy_last_5"]    = _accuracy_window(skill_events, n=5)
    feats["accuracy_last_10"]   = _accuracy_window(skill_events, n=10)
    feats["accuracy_all_time"]  = _accuracy_window(skill_events, n=len(skill_events))
    feats["accuracy_trend"]     = _trend(
        [float(e.get("correct", 0)) for e in skill_events[-10:]]
    )

    # --- Response time features ---
    rt_values = [e.get("response_time_ms", 3000) for e in skill_events if e.get("response_time_ms") is not None]
    feats["response_time_mean"]   = _safe_mean(rt_values)
    feats["response_time_last_3"] = _safe_mean(rt_values[-3:]) if rt_values else 3000.0
    feats["response_time_trend"]  = _trend(rt_values[-10:])
    feats["response_time_cv"]     = _coeff_variation(rt_values)

    # --- Retry / hint features ---
    feats["retry_count_mean"]   = _safe_mean([e.get("retry_count", 0) for e in skill_events])
    feats["hint_usage_rate"]    = _rate([e.get("hint_used", False) for e in skill_events])
    feats["hint_dependency_trend"] = _trend(
        [float(e.get("hint_used", False)) for e in skill_events[-10:]]
    )

    # --- Streak features ---
    feats["current_streak"]     = _current_streak(skill_events)
    feats["max_streak"]         = _max_streak(skill_events)
    feats["streak_breaks"]      = _count_streak_breaks(skill_events)

    # --- Error burst features ---
    feats["error_burst_count"]  = _error_burst_count(skill_events, burst_len=3)
    feats["consec_errors_now"]  = _consecutive_errors_now(skill_events)

    # --- Modality success rates ---
    feats["visual_success_rate"]       = _modality_success(skill_events, "visual")
    feats["auditory_success_rate"]     = _modality_success(skill_events, "auditory")
    feats["kinesthetic_success_rate"]  = _modality_success(skill_events, "kinesthetic")

    # --- Attempt counts ---
    feats["attempts_total"]     = len(skill_events)
    feats["attempts_recent"]    = len([e for e in all_recent if e.get("skill") == skill])
    feats["has_min_attempts"]   = float(len(skill_events) >= 5)   # cold start flag

    # --- Prior mastery from profile ---
    feats["prior_mastery_prob"] = child_profile.get("skill_mastery_probs", {}).get(skill, 0.1)

    # --- Child context ---
    feats["age_months"]         = child_profile.get("age_months", 60)
    feats["age_band_encoded"]   = _encode_age_band(child_profile.get("age_months", 60))
    feats["sessions_completed"] = child_profile.get("sessions_completed", 0)
    feats["preferred_modality_encoded"] = MODALITY_ENCODING.get(
        child_profile.get("preferred_modality", DEFAULT_MODALITY), 0
    )

    # --- Time features ---
    feats["days_since_last_session"] = child_profile.get("days_since_last_session", 0)

    return feats


def extract_engagement_features(
    events: list[dict],
    child_profile: dict,
    session_elapsed_seconds: float,
    session_allowed_seconds: float,
) -> dict:
    """
    Build the feature vector for the Engagement / Fatigue Classifier.
    Session-level features derived from the most recent window of events.
    """
    recent = events[-SESSION_WINDOW_SIZE:] if len(events) > SESSION_WINDOW_SIZE else events

    feats = {}

    # --- Response time features ---
    rt_all    = [e.get("response_time_ms", 3000) for e in recent if e.get("response_time_ms") is not None]
    rt_last3  = rt_all[-3:] if len(rt_all) >= 3 else rt_all
    rt_first3 = rt_all[:3] if len(rt_all) >= 3 else rt_all

    feats["response_time_mean"]       = _safe_mean(rt_all)
    feats["response_time_last3_mean"] = _safe_mean(rt_last3)
    feats["response_time_trend"]      = _trend(rt_all)           # positive = slowing = fatigue
    feats["response_time_zscore"]     = _zscore_last(rt_all)     # how abnormal is latest RT?
    feats["response_time_cv"]         = _coeff_variation(rt_all) # variability

    # --- Error / retry features ---
    feats["error_rate_last_5"]   = 1.0 - _accuracy_window(recent, n=5)
    feats["error_rate_last_10"]  = 1.0 - _accuracy_window(recent, n=10)
    feats["retry_rate_session"]  = _rate([e.get("retry_count", 0) > 0 for e in recent])
    feats["consec_errors_now"]   = _consecutive_errors_now(recent)
    feats["error_burst_flag"]    = float(_error_burst_count(recent, burst_len=3) > 0)

    # --- Pause / inactivity features ---
    pause_durations    = [e.get("pause_duration_ms", 0) for e in recent if e.get("pause_duration_ms") is not None]
    inactivity_values  = [e.get("inactivity_duration_ms", 0) for e in recent if e.get("inactivity_duration_ms") is not None]
    feats["pause_duration_mean"]       = _safe_mean(pause_durations)
    feats["pause_duration_max"]        = max(pause_durations) if pause_durations else 0.0
    feats["inactivity_total_ms"]       = sum(inactivity_values)
    feats["inactivity_event_count"]    = sum(1 for v in inactivity_values if v > 5000)  # >5s gaps

    # --- Hint features ---
    feats["hint_usage_rate_session"]   = _rate([e.get("hint_used", False) for e in recent])
    feats["hint_rate_trend"]           = _trend([float(e.get("hint_used", False)) for e in recent[-10:]])

    # --- Streak features ---
    feats["streak_break_count"]        = _count_streak_breaks(recent)
    feats["current_streak"]            = _current_streak(recent)

    # --- Session pace / time features ---
    feats["session_elapsed_seconds"]   = session_elapsed_seconds
    feats["session_duration_ratio"]    = min(session_elapsed_seconds / max(session_allowed_seconds, 1.0), 1.0)
    feats["tasks_per_minute"]          = len(recent) / max(session_elapsed_seconds / 60.0, 0.5)

    # --- Task switching ---
    task_types = [e.get("task_type", "unknown") for e in recent]
    feats["task_switch_count"] = sum(1 for i in range(1, len(task_types)) if task_types[i] != task_types[i - 1])

    # --- Reward response ---
    reward_responses = [e.get("reward_response", 0.5) for e in recent if e.get("reward_response") is not None]
    feats["reward_response_mean"] = _safe_mean(reward_responses)

    # --- Time to first response (attention proxy) ---
    ttfr_values = [e.get("time_to_first_response_ms", 2000) for e in recent if e.get("time_to_first_response_ms") is not None]
    feats["time_to_first_response_mean"] = _safe_mean(ttfr_values)
    feats["time_to_first_response_trend"] = _trend(ttfr_values[-10:])

    # --- Child context ---
    feats["age_months"]           = child_profile.get("age_months", 60)
    feats["age_band_encoded"]     = _encode_age_band(child_profile.get("age_months", 60))
    feats["sessions_completed"]   = child_profile.get("sessions_completed", 0)

    return feats


def get_skill_feature_names() -> list[str]:
    """Return ordered list of feature names for Skill Mastery model."""
    dummy_profile = {"age_months": 60, "skill_mastery_probs": {}, "sessions_completed": 0,
                     "preferred_modality": "visual", "days_since_last_session": 0}
    dummy_events  = []
    sample = extract_skill_features(dummy_events, SKILLS[0], dummy_profile)
    return list(sample.keys())


def get_engagement_feature_names() -> list[str]:
    """Return ordered list of feature names for Engagement model."""
    dummy_profile = {"age_months": 60, "sessions_completed": 0}
    sample = extract_engagement_features([], dummy_profile, 600.0, 1200.0)
    return list(sample.keys())


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _accuracy_window(events: list[dict], n: int) -> float:
    """Mean correctness over last n events. Returns 0.5 if empty (neutral prior)."""
    window = events[-n:] if len(events) >= n else events
    if not window:
        return 0.5
    return sum(float(e.get("correct", 0)) for e in window) / len(window)


def _safe_mean(values: list) -> float:
    if not values:
        return 0.0
    return sum(float(v) for v in values) / len(values)


def _rate(booleans: list) -> float:
    """Proportion of True / truthy values."""
    if not booleans:
        return 0.0
    return sum(1.0 for v in booleans if v) / len(booleans)


def _trend(values: list[float]) -> float:
    """
    Linear regression slope over the values (normalized by count).
    Positive = increasing, Negative = decreasing.
    Returns 0.0 if fewer than 2 data points.
    """
    n = len(values)
    if n < 2:
        return 0.0
    x_mean = (n - 1) / 2.0
    y_mean = _safe_mean(values)
    num = sum((i - x_mean) * (v - y_mean) for i, v in enumerate(values))
    den = sum((i - x_mean) ** 2 for i in range(n))
    return num / den if den != 0 else 0.0


def _coeff_variation(values: list[float]) -> float:
    """Coefficient of variation = std / mean. 0.0 if insufficient data."""
    if len(values) < 2:
        return 0.0
    mean = _safe_mean(values)
    if mean == 0:
        return 0.0
    variance = sum((v - mean) ** 2 for v in values) / len(values)
    return math.sqrt(variance) / mean


def _zscore_last(values: list[float]) -> float:
    """Z-score of the last value relative to the rest."""
    if len(values) < 3:
        return 0.0
    prior = values[:-1]
    mean  = _safe_mean(prior)
    std   = math.sqrt(sum((v - mean) ** 2 for v in prior) / len(prior))
    if std == 0:
        return 0.0
    return (values[-1] - mean) / std


def _current_streak(events: list[dict]) -> int:
    """Number of consecutive correct answers at the tail of events."""
    streak = 0
    for e in reversed(events):
        if e.get("correct", False):
            streak += 1
        else:
            break
    return streak


def _max_streak(events: list[dict]) -> int:
    """Longest consecutive correct run in events."""
    max_s, cur = 0, 0
    for e in events:
        if e.get("correct", False):
            cur += 1
            max_s = max(max_s, cur)
        else:
            cur = 0
    return max_s


def _count_streak_breaks(events: list[dict]) -> int:
    """Count how many times a correct run was broken by an error."""
    breaks, in_streak = 0, False
    for e in events:
        if e.get("correct", False):
            in_streak = True
        elif in_streak:
            breaks += 1
            in_streak = False
    return breaks


def _error_burst_count(events: list[dict], burst_len: int = 3) -> int:
    """Count occurrences of burst_len consecutive errors."""
    count, run = 0, 0
    for e in events:
        if not e.get("correct", True):
            run += 1
            if run == burst_len:
                count += 1
                run = 0
        else:
            run = 0
    return count


def _consecutive_errors_now(events: list[dict]) -> int:
    """Number of consecutive errors at the very tail of events."""
    count = 0
    for e in reversed(events):
        if not e.get("correct", True):
            count += 1
        else:
            break
    return count


def _modality_success(events: list[dict], modality: str) -> float:
    """Success rate for a specific modality. Returns 0.5 if no data."""
    mod_events = [e for e in events if e.get("modality") == modality]
    if not mod_events:
        return 0.5
    return sum(float(e.get("correct", 0)) for e in mod_events) / len(mod_events)


def _encode_age_band(age_months: int) -> int:
    """
    0 = toddler (2-4 y), 1 = preschool (4-6 y),
    2 = early_primary (6-8 y), 3 = primary (8-11 y)
    """
    age_years = age_months / 12.0
    if age_years < 4:
        return 0
    elif age_years < 6:
        return 1
    elif age_years < 8:
        return 2
    return 3


# Global constant for external use
SESSION_WINDOW_SIZE = 20
