# =============================================================================
# engine/profile_builder.py
# Formula-based starter profile creation from parent onboarding data.
# Zero ML. All logic is deterministic.
# =============================================================================

import uuid
from datetime import datetime, timezone
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    SKILLS,
    AGE_BANDS,
    AGE_BAND_PRIORS,
    AGE_BAND_START_DIFFICULTY,
    AGE_SESSION_CAPS,
    INTEREST_TO_MODALITY,
    DIFFICULTY_LABELS,
    DEFAULT_MODALITY,
    MODALITIES,
    SKILL_PREREQUISITES,
    SKILL_DISPLAY_NAMES,
)


def build_starter_profile(onboarding_data: dict) -> dict:
    """
    Convert parent onboarding answers into a complete ChildProfile JSON.

    Parameters
    ----------
    onboarding_data : dict with keys:
        child_name              : str
        child_age_years         : float
        language                : str (e.g. "en", "hi", "es")
        interests               : list[str]
        goals                   : list[str]  (e.g. ["learn phonics", "improve focus"])
        sensitivities           : list[str]  (e.g. ["loud_sounds", "bright_lights"])
        allowed_session_minutes : int

    Returns
    -------
    ChildProfile dict (matches schemas/starter_profile_schema.json)
    """
    name        = onboarding_data.get("child_name", "Child")
    age_years   = float(onboarding_data.get("child_age_years", 5.0))
    age_months  = int(age_years * 12)
    language    = onboarding_data.get("language", "en")
    interests   = onboarding_data.get("interests", [])
    goals       = onboarding_data.get("goals", [])
    sensitivities = onboarding_data.get("sensitivities", [])
    allowed_min = int(onboarding_data.get("allowed_session_minutes", 15))

    age_band    = _get_age_band(age_years)
    priors      = AGE_BAND_PRIORS[age_band]
    start_diff  = AGE_BAND_START_DIFFICULTY[age_band]
    rec_cap     = AGE_SESSION_CAPS[age_band]
    session_cap = min(allowed_min, rec_cap)

    preferred_modality = _infer_preferred_modality(interests, sensitivities)
    start_difficulty   = _adjust_start_difficulty(start_diff, goals, age_years)

    # Initial skill estimates = age priors adjusted for stated goals
    initial_skill_estimates = _compute_initial_estimates(priors, goals, age_band)

    # Compute unlocked skills (prerequisites satisfied)
    unlocked = _get_unlocked_skills(initial_skill_estimates)

    # Recommended first session plan
    first_session_plan = _build_first_session_plan(
        initial_skill_estimates, preferred_modality, start_difficulty, session_cap, age_band
    )

    now = datetime.now(timezone.utc).isoformat()

    profile = {
        "child_id":                str(uuid.uuid4()),
        "name":                    name,
        "age_months":              age_months,
        "age_band":                age_band,
        "language":                language,
        "interests":               interests,
        "goals":                   goals,
        "sensitivities":           sensitivities,
        "sessions_completed":      0,
        "preferred_modality":      preferred_modality,
        "allowed_session_minutes": session_cap,
        "current_difficulty_per_skill": {s: start_difficulty for s in SKILLS},
        "skill_mastery_probs":     initial_skill_estimates,
        "unlocked_skills":         unlocked,
        "days_since_last_session": 0,
        "flags": {
            "new_user":          True,
            "needs_calibration": True,
            "cold_start":        True,
        },
        "safe_difficulty_range":   _safe_difficulty_range(age_band),
        "initial_skill_estimates": {
            s: {
                "mastery_prob":   round(initial_skill_estimates[s], 4),
                "mastery_class":  _classify_mastery(initial_skill_estimates[s]),
                "source":         "age_band_prior",
                "confidence":     0.40,
            }
            for s in SKILLS
        },
        "recommended_first_session_plan": first_session_plan,
        "created_at":  now,
        "updated_at":  now,
    }

    return profile


# ---------------------------------------------------------------------------
# Sub-routines
# ---------------------------------------------------------------------------

def _get_age_band(age_years: float) -> str:
    for band, (lo, hi) in AGE_BANDS.items():
        if lo <= age_years < hi:
            return band
    # Above defined range → use highest band
    return "primary"


def _infer_preferred_modality(interests: list, sensitivities: list) -> str:
    """
    Map interests to modality preferences.
    Sensitivities can veto certain modalities.
    """
    modality_votes = {m: 0 for m in MODALITIES}

    for interest in interests:
        mapped = INTEREST_TO_MODALITY.get(interest.lower().strip())
        if mapped:
            modality_votes[mapped] += 2

    # Veto logic
    vetoed = set()
    if "loud_sounds" in sensitivities or "sound_sensitive" in sensitivities:
        vetoed.add("auditory")
    if "light_sensitive" in sensitivities or "bright_lights" in sensitivities:
        vetoed.add("visual")
    if "movement_difficulty" in sensitivities:
        vetoed.add("kinesthetic")

    # Pick highest voted non-vetoed modality
    valid_votes = {m: v for m, v in modality_votes.items() if m not in vetoed}
    if not valid_votes or max(valid_votes.values()) == 0:
        # Default per age: toddlers like visual, older kids can do mixed
        remaining = [m for m in MODALITIES if m not in vetoed]
        return remaining[0] if remaining else DEFAULT_MODALITY

    return max(valid_votes, key=lambda m: valid_votes[m])


def _adjust_start_difficulty(base_difficulty: int, goals: list, age_years: float) -> int:
    """
    Fine-tune starting difficulty based on goals.
    Parents who mention "challenge" or "advanced" bump up by 1.
    Parents who mention "struggling", "behind", "slow" reduce by 1.
    """
    goal_text = " ".join(goals).lower()
    if any(w in goal_text for w in ["challenge", "advanced", "gifted", "ahead"]):
        return min(base_difficulty + 1, 5)
    if any(w in goal_text for w in ["struggling", "behind", "slow", "difficulty", "help"]):
        return max(base_difficulty - 1, 1)
    return base_difficulty


def _compute_initial_estimates(
    priors: dict,
    goals: list,
    age_band: str,
) -> dict:
    """
    Start from age-band priors.
    Boost skills explicitly mentioned in goals by +0.05 (small — goals suggest focus, not mastery).
    """
    estimates = dict(priors)
    goal_text = " ".join(goals).lower()

    goal_skill_keywords = {
        "phonics":               ["phonics", "reading", "sounds", "letters"],
        "letter_recognition":    ["alphabet", "letters", "letter"],
        "counting":              ["count", "numbers", "math", "numeracy"],
        "number_order":          ["ordering", "sequence", "number order"],
        "shape_recognition":     ["shapes", "geometry"],
        "color_sorting":         ["colors", "colour", "sorting"],
        "hand_eye_coordination": ["coordination", "motor", "drawing", "writing"],
        "attention_retention":   ["focus", "attention", "concentration"],
        "memory_recall":         ["memory", "recall", "remember"],
        "social_response":       ["social", "friends", "interaction"],
        "emotional_regulation":  ["emotions", "feelings", "calm", "emotional"],
    }

    for skill, keywords in goal_skill_keywords.items():
        if any(kw in goal_text for kw in keywords):
            estimates[skill] = min(estimates[skill] + 0.05, 0.95)

    return {s: round(estimates[s], 4) for s in SKILLS}


def _get_unlocked_skills(mastery_probs: dict) -> list:
    """Return skills whose prerequisites are satisfied."""
    from config import PREREQUISITE_GATE_THRESHOLD

    unlocked = []
    for skill, prereqs in SKILL_PREREQUISITES.items():
        if all(mastery_probs.get(p, 0) >= PREREQUISITE_GATE_THRESHOLD for p in prereqs):
            unlocked.append(skill)
    return sorted(unlocked)


def _safe_difficulty_range(age_band: str) -> dict:
    from config import AGE_BAND_START_DIFFICULTY, DIFFICULTY_LABELS

    start = AGE_BAND_START_DIFFICULTY[age_band]
    lo    = max(1, start - 1)
    hi    = min(5, start + 2)
    return {
        "min": lo,
        "max": hi,
        "min_label": DIFFICULTY_LABELS[lo],
        "max_label": DIFFICULTY_LABELS[hi],
    }


def _build_first_session_plan(
    skill_estimates: dict,
    preferred_modality: str,
    difficulty: int,
    session_cap_minutes: int,
    age_band: str,
) -> dict:
    """
    Recommend a first session: pick the 2 most-appropriate skills,
    sequence them with warm-up → main → cool-down structure.
    """
    # Sort skills by "how ready to learn" — moderate prior, unlocked
    scored = {}
    for skill, prob in skill_estimates.items():
        prereqs = SKILL_PREREQUISITES.get(skill, [])
        prereq_met = all(skill_estimates.get(p, 0) >= 0.35 for p in prereqs)
        if prereq_met:
            # Best starting zone: not too easy (prob > 0.3), not too hard (prob < 0.75)
            readiness = 1.0 - abs(prob - 0.50) * 2
            scored[skill] = readiness

    top_skills = sorted(scored, key=lambda s: scored[s], reverse=True)[:3]

    return {
        "structure":         ["warm_up", "main_activity", "cool_down"],
        "warm_up_skill":     top_skills[0] if top_skills else "color_sorting",
        "main_skill":        top_skills[1] if len(top_skills) > 1 else "counting",
        "cool_down_skill":   top_skills[2] if len(top_skills) > 2 else "shape_recognition",
        "starting_difficulty": difficulty,
        "preferred_modality": preferred_modality,
        "session_minutes":   session_cap_minutes,
        "tasks_per_skill":   max(3, session_cap_minutes // 3),
        "hint_mode":         "generous",
        "reward_frequency":  "high",
        "note":              "First session is a calibration session — the app will observe and adjust."
    }


def _classify_mastery(prob: float) -> str:
    if prob < 0.40:
        return "low"
    elif prob < 0.70:
        return "developing"
    return "high"
