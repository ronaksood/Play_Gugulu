# =============================================================================
# engine/formula_engine.py
# THE DECISION LAYER — completely deterministic, no ML.
#
# Takes the two ML model outputs and applies rule-based formulas to produce:
#   - next skill to focus on
#   - difficulty level
#   - modality
#   - hint intensity
#   - reward type
#   - session length adjustment
#   - end-session signal
#   - offline activity suggestion
#   - parent tip
# =============================================================================

import random
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    SKILLS,
    SKILL_PREREQUISITES,
    DIFFICULTY_LEVELS,
    DIFFICULTY_MIN,
    DIFFICULTY_MAX,
    DIFFICULTY_LABELS,
    MODALITIES,
    FATIGUE_MODALITY,
    DEFAULT_MODALITY,
    MASTERY_LOW_THRESHOLD,
    MASTERY_HIGH_THRESHOLD,
    FATIGUE_CONFIDENCE_HARD_STOP,
    FATIGUE_CONFIDENCE_SOFT_ALERT,
    HINT_ERROR_BURST_TRIGGER,
    HINT_COOLDOWN_EVENTS,
    SESSION_NEAR_END_RATIO,
    AGE_SESSION_CAPS,
    PREREQUISITE_GATE_THRESHOLD,
    REWARD_TYPES,
    SKILL_DISPLAY_NAMES,
    SKILL_CATEGORIES,
)


def decide_next_action(
    child_profile: dict,
    skill_graph: dict,
    engagement_result: dict,
    session_events: list[dict],
    session_elapsed_seconds: float,
) -> dict:
    """
    Core formula engine. Given ML model outputs, compute the next action.

    Parameters
    ----------
    child_profile       : updated ChildProfile dict (post skill graph update)
    skill_graph         : output of update_skill_graph()
    engagement_result   : output of EngagementClassifier.predict()
    session_events      : full list of events this session
    session_elapsed_seconds : seconds elapsed in current session

    Returns
    -------
    next_activity dict (matches schemas/next_activity_schema.json)
    """
    child_id   = child_profile.get("child_id", "unknown")
    session_id = child_profile.get("session_id", "unknown")
    age_band   = child_profile.get("age_band", "preschool")

    mastery_probs   = child_profile.get("skill_mastery_probs", {})
    skill_nodes_map = {n["skill_name"]: n for n in skill_graph.get("skill_nodes", [])}
    allowed_seconds = child_profile.get("allowed_session_minutes", 15) * 60.0
    unlocked_skills = child_profile.get("unlocked_skills", SKILLS)

    engagement_state = engagement_result.get("engagement_state", "engaged")
    fatigue_risk     = engagement_result.get("fatigue_risk", 0.0)
    alert_level      = engagement_result.get("alert_level", "none")

    # -----------------------------------------------------------------------
    # 1. Should the session end?
    # -----------------------------------------------------------------------
    end_session, end_reason = _check_session_end(
        engagement_result, session_elapsed_seconds, allowed_seconds, len(session_events)
    )

    if end_session:
        return _build_end_session_output(child_id, session_id, end_reason, age_band, child_profile)

    # -----------------------------------------------------------------------
    # 2. Select next skill
    # -----------------------------------------------------------------------
    next_skill = _select_next_skill(
        mastery_probs, unlocked_skills, skill_nodes_map, session_events, engagement_state
    )
    skill_mastery = mastery_probs.get(next_skill, 0.3)
    skill_node    = skill_nodes_map.get(next_skill, {})
    skill_confidence = skill_node.get("confidence", 0.40)

    # -----------------------------------------------------------------------
    # 3. Select difficulty
    # -----------------------------------------------------------------------
    current_difficulty = child_profile.get("current_difficulty_per_skill", {}).get(next_skill, 2)
    next_difficulty    = _select_difficulty(
        skill_mastery, skill_confidence, current_difficulty,
        session_events, next_skill, engagement_state, fatigue_risk, age_band
    )
    # Update profile with new difficulty
    child_profile.setdefault("current_difficulty_per_skill", {})[next_skill] = next_difficulty

    # -----------------------------------------------------------------------
    # 4. Select modality
    # -----------------------------------------------------------------------
    next_modality = _select_modality(
        child_profile, next_skill, engagement_state, fatigue_risk, session_events
    )

    # -----------------------------------------------------------------------
    # 5. Compute hint intensity
    # -----------------------------------------------------------------------
    hint_intensity = _compute_hint_intensity(
        skill_mastery, session_events, next_skill, engagement_state
    )

    # -----------------------------------------------------------------------
    # 6. Select reward type
    # -----------------------------------------------------------------------
    reward_type = _select_reward(session_events, skill_mastery, skill_node)

    # -----------------------------------------------------------------------
    # 7. Session length adjustment
    # -----------------------------------------------------------------------
    adjusted_remaining = _adjust_session_remaining(
        allowed_seconds, session_elapsed_seconds, fatigue_risk, alert_level
    )

    # -----------------------------------------------------------------------
    # 8. Parent tip (rule-based text)
    # -----------------------------------------------------------------------
    parent_tip = _get_parent_tip(next_skill, skill_mastery, next_modality, engagement_state, age_band)

    # -----------------------------------------------------------------------
    # 9. Build reasoning string (interpretable)
    # -----------------------------------------------------------------------
    reason = _build_reason(
        next_skill, skill_mastery, skill_confidence, next_difficulty,
        next_modality, engagement_state, fatigue_risk
    )

    return {
        "child_id":                  child_id,
        "session_id":                session_id,
        "next_task_id":              f"task_{next_skill}_{next_difficulty}_{next_modality}",
        "next_skill_focus":          next_skill,
        "next_skill_display":        SKILL_DISPLAY_NAMES.get(next_skill, next_skill),
        "next_modality":             next_modality,
        "next_difficulty":           next_difficulty,
        "next_difficulty_label":     DIFFICULTY_LABELS[next_difficulty],
        "hint_intensity":            hint_intensity,
        "reward_type":               reward_type,
        "session_length_adjustment": {
            "remaining_seconds":     round(adjusted_remaining),
            "action":                alert_level if alert_level != "none" else "maintain",
        },
        "end_session":               False,
        "parent_tip":                parent_tip,
        "reason":                    reason,
        "mastery_context": {
            "skill":       next_skill,
            "mastery_prob": round(skill_mastery, 4),
            "mastery_class": _classify_mastery(skill_mastery),
            "confidence":   round(skill_confidence, 4),
        },
    }


# ---------------------------------------------------------------------------
# Decision sub-functions
# ---------------------------------------------------------------------------

def _check_session_end(
    engagement_result: dict,
    elapsed: float,
    allowed: float,
    n_events: int,
) -> tuple[bool, str]:
    fatigue_risk = engagement_result.get("fatigue_risk", 0.0)
    duration_ratio = elapsed / max(allowed, 1.0)

    if fatigue_risk >= FATIGUE_CONFIDENCE_HARD_STOP:
        return True, "fatigue_detected"
    if duration_ratio >= SESSION_NEAR_END_RATIO:
        return True, "session_time_limit"
    if n_events > 0 and elapsed > 3600:   # absolute safety cap: 60 min
        return True, "absolute_time_cap"
    return False, ""


def _select_next_skill(
    mastery_probs: dict,
    unlocked_skills: list,
    skill_nodes_map: dict,
    session_events: list,
    engagement_state: str,
) -> str:
    """
    Priority order:
    1. If fatigued → pick highest-mastery unlocked skill (easiest win = re-engagement)
    2. Otherwise:
       a. 70% chance: pick lowest-mastery unlocked skill (zone of proximal development)
       b. 30% chance: pick a "consolidation" skill near mastery threshold (reinforce)

    Tiebreaker: prefer skills not seen in the last 3 events (variety).
    """
    if not unlocked_skills:
        return SKILLS[0]

    recent_skills = [e.get("skill") for e in session_events[-3:]]
    unlocked      = [s for s in unlocked_skills if mastery_probs.get(s, 0.1) is not None]

    if not unlocked:
        return unlocked_skills[0]

    if engagement_state == "fatigued":
        # High mastery = comfort zone = re-engaging
        scored = sorted(unlocked, key=lambda s: mastery_probs.get(s, 0.1), reverse=True)
        return scored[0]

    # Prefer variety (not recently seen)
    fresh = [s for s in unlocked if s not in recent_skills] or unlocked

    # Zone of Proximal Development: 0.40 ≤ mastery ≤ 0.70 is ideal
    zpd_skills = [s for s in fresh if MASTERY_LOW_THRESHOLD <= mastery_probs.get(s, 0.1) <= MASTERY_HIGH_THRESHOLD]

    # 70/30 split: growth vs consolidation
    roll = random.random()
    if zpd_skills and roll < 0.50:
        # Pick ZPD skill that has a regression flag (highest reinforcement need)
        regressing = [
            s for s in zpd_skills
            if skill_nodes_map.get(s, {}).get("regression_flagged", False)
        ]
        return regressing[0] if regressing else zpd_skills[0]

    if roll < 0.70:
        # Growth: lowest mastery unlocked skill
        return min(fresh, key=lambda s: mastery_probs.get(s, 0.1))

    # Consolidation: skill close to mastery boundary
    return min(
        fresh,
        key=lambda s: abs(mastery_probs.get(s, 0.1) - MASTERY_HIGH_THRESHOLD)
    )


def _select_difficulty(
    skill_mastery: float,
    skill_confidence: float,
    current_difficulty: int,
    session_events: list,
    skill: str,
    engagement_state: str,
    fatigue_risk: float,
    age_band: str,
) -> int:
    """
    Core difficulty formula:

    mastery_prob < 0.40              → decrease difficulty by 1
    0.40 ≤ mastery_prob < 0.70       → maintain current difficulty
    mastery_prob ≥ 0.70              → increase difficulty by 1

    Modifiers:
    - If model confidence < 0.55     → stay at current (uncertainty buffer)
    - If engagement_state == fatigued → decrease by 1 (extra)
    - If recent 3 tasks all correct  → allow +1 regardless of mastery prob
    - If recent 3 tasks all wrong    → force -1 regardless
    - Clamp within age-band safe range
    """
    from config import AGE_BAND_START_DIFFICULTY

    # Uncertainty buffer
    if skill_confidence < 0.55:
        return _clamp_difficulty(current_difficulty, age_band)

    # Base formula
    if skill_mastery < MASTERY_LOW_THRESHOLD:
        new_diff = current_difficulty - 1
    elif skill_mastery >= MASTERY_HIGH_THRESHOLD:
        new_diff = current_difficulty + 1
    else:
        new_diff = current_difficulty   # ZPD zone: maintain

    # Override: recent performance check
    recent_skill_events = [e for e in session_events[-5:] if e.get("skill") == skill]
    if len(recent_skill_events) >= 3:
        last_3_correct = all(e.get("correct", False) for e in recent_skill_events[-3:])
        last_3_wrong   = not any(e.get("correct", False) for e in recent_skill_events[-3:])
        if last_3_correct:
            new_diff = max(new_diff, current_difficulty + 1)
        if last_3_wrong:
            new_diff = min(new_diff, current_difficulty - 1)

    # Fatigue modifier
    if engagement_state == "fatigued" or fatigue_risk >= FATIGUE_CONFIDENCE_SOFT_ALERT:
        new_diff -= 1

    return _clamp_difficulty(new_diff, age_band)


def _select_modality(
    child_profile: dict,
    skill: str,
    engagement_state: str,
    fatigue_risk: float,
    session_events: list,
) -> str:
    """
    Modality selection formula:

    1. If fatigued → kinesthetic (movement re-engages)
    2. If attention is dropping (fatigue_risk > soft alert) → switch from static to kinesthetic
    3. If child has strong modality for this skill (success_rate > 0.80) → use that modality
    4. Otherwise → use preferred_modality from profile
    """
    preferred = child_profile.get("preferred_modality", DEFAULT_MODALITY)

    if engagement_state == "fatigued":
        return FATIGUE_MODALITY

    if fatigue_risk >= FATIGUE_CONFIDENCE_SOFT_ALERT and preferred in ("visual",):
        return FATIGUE_MODALITY

    # Detect strongest modality for this skill from recent events
    skill_events = [e for e in session_events if e.get("skill") == skill and e.get("modality")]
    if len(skill_events) >= 5:
        modality_success = {}
        for m in MODALITIES:
            m_events = [e for e in skill_events if e.get("modality") == m]
            if len(m_events) >= 2:
                rate = sum(1 for e in m_events if e.get("correct", False)) / len(m_events)
                modality_success[m] = rate

        if modality_success:
            best_modality = max(modality_success, key=lambda m: modality_success[m])
            if modality_success[best_modality] >= 0.80:
                return best_modality

    return preferred


def _compute_hint_intensity(
    skill_mastery: float,
    session_events: list,
    skill: str,
    engagement_state: str,
) -> str:
    """
    Hint intensity formula:

    mastery_prob < 0.30                → "high"   (step-by-step guide)
    0.30 ≤ mastery_prob < 0.50         → "medium" (single hint)
    mastery_prob ≥ 0.50                → "low"    (encourage only)

    Override:
    - If 3+ consecutive errors on this skill → force "high"
    - If fatigued → bump up one level (child needs more support)
    """
    # Consecutive errors override
    skill_events  = [e for e in session_events if e.get("skill") == skill]
    consec_errors = 0
    for e in reversed(skill_events):
        if not e.get("correct", True):
            consec_errors += 1
        else:
            break

    if consec_errors >= HINT_ERROR_BURST_TRIGGER:
        return "high"

    # Fatigue bump
    if engagement_state == "fatigued":
        if skill_mastery < 0.50:
            return "high"
        return "medium"

    # Mastery-based
    if skill_mastery < 0.30:
        return "high"
    elif skill_mastery < 0.50:
        return "medium"
    return "low"


def _select_reward(
    session_events: list,
    skill_mastery: float,
    skill_node: dict,
) -> str:
    """
    Reward selection formula:

    - Current streak >= 5         → "celebration"
    - Mastery just crossed 0.70  → "progress_badge"
    - Recovering from regression → "encouragement_star"
    - Mastery newly improving    → "streak_flame"
    - Default                    → "sparkle"
    """
    # Streak check
    streak = 0
    for e in reversed(session_events):
        if e.get("correct", False):
            streak += 1
        else:
            break

    if streak >= 5:
        return "celebration"

    # Milestone crossing
    prev_mastery  = skill_node.get("previous_mastery", skill_mastery)
    if prev_mastery < MASTERY_HIGH_THRESHOLD <= skill_mastery:
        return "progress_badge"

    # Regression recovery
    if skill_node.get("regression_flagged", False) and skill_node.get("trend") == "improving":
        return "encouragement_star"

    # General improvement
    if skill_node.get("trend") == "improving":
        return "streak_flame"

    return "sparkle"


def _adjust_session_remaining(
    allowed_seconds: float,
    elapsed_seconds: float,
    fatigue_risk: float,
    alert_level: str,
) -> float:
    """
    Remaining session time formula:

    remaining = allowed - elapsed
    If alert_level == "soft" → remaining * 0.80 (shorten by 20%)
    If fatigue_risk > SOFT_ALERT → cap remaining at 5 minutes
    Otherwise → maintain planned remaining
    """
    remaining = max(0.0, allowed_seconds - elapsed_seconds)

    if alert_level == "soft" or fatigue_risk >= FATIGUE_CONFIDENCE_SOFT_ALERT:
        remaining = remaining * 0.80

    return min(remaining, 5 * 60) if fatigue_risk >= FATIGUE_CONFIDENCE_SOFT_ALERT else remaining


def _get_parent_tip(
    skill: str,
    mastery_prob: float,
    modality: str,
    engagement_state: str,
    age_band: str,
) -> str:
    """
    Rule-based parent tip text.
    No ML — pure template selection based on observable conditions.
    """
    display = SKILL_DISPLAY_NAMES.get(skill, skill)

    if engagement_state == "fatigued":
        return f"Today's session was great — try a short movement break before the next one."

    if mastery_prob < MASTERY_LOW_THRESHOLD:
        age_tip = {
            "toddler":       "Use everyday objects to practice — toys, household items work great.",
            "preschool":     "Short, playful practice moments work best — even 5 minutes helps.",
            "early_primary": "Try making it a game at home — repetition through play is powerful.",
            "primary":       "A little daily practice goes a long way with this skill.",
        }
        return f"{display} needs more practice. " + age_tip.get(age_band, "Short daily sessions help.")

    if mastery_prob >= MASTERY_HIGH_THRESHOLD:
        return f"{display} is going really well! Try introducing it in new contexts at home."

    if modality == "kinesthetic":
        return f"Your child learns {display} best through movement. Try active games at home."
    elif modality == "auditory":
        return f"Songs and rhymes are a great way to reinforce {display} at home."
    elif modality == "story":
        return f"Storybooks and bedtime reading will reinforce {display} naturally."

    return f"{display} is developing well. Keep up the regular practice!"


def _build_reason(
    skill: str,
    mastery_prob: float,
    confidence: float,
    difficulty: int,
    modality: str,
    engagement_state: str,
    fatigue_risk: float,
) -> str:
    """Build a short, interpretable reason string for the next action."""
    parts = []

    mastery_class = _classify_mastery(mastery_prob)
    parts.append(f"{skill} selected ({mastery_class} mastery, {int(mastery_prob*100)}%)")

    if confidence < 0.55:
        parts.append("difficulty held steady due to low confidence")
    elif mastery_prob < MASTERY_LOW_THRESHOLD:
        parts.append(f"difficulty reduced to {DIFFICULTY_LABELS[difficulty]}")
    elif mastery_prob >= MASTERY_HIGH_THRESHOLD:
        parts.append(f"difficulty increased to {DIFFICULTY_LABELS[difficulty]}")

    if engagement_state == "fatigued":
        parts.append(f"switched to {modality} mode due to fatigue")
    elif fatigue_risk >= FATIGUE_CONFIDENCE_SOFT_ALERT:
        parts.append(f"session shortened — attention signal detected")

    return "; ".join(parts) + "."


# ---------------------------------------------------------------------------
# End-session output
# ---------------------------------------------------------------------------

def _build_end_session_output(
    child_id: str,
    session_id: str,
    reason: str,
    age_band: str,
    child_profile: dict,
) -> dict:
    """Victory moment — called when session should end."""
    mastery_probs = child_profile.get("skill_mastery_probs", {})
    top_skill     = max(mastery_probs, key=mastery_probs.get, default="counting")

    offline_tips = {
        "fatigue_detected":  "Rest and come back tomorrow — short breaks improve learning.",
        "session_time_limit": f"Great session! Try counting objects around the house today.",
        "absolute_time_cap":  "Excellent effort! Time for a movement break.",
    }

    return {
        "child_id":              child_id,
        "session_id":            session_id,
        "next_task_id":          "victory_moment",
        "next_skill_focus":      top_skill,
        "next_skill_display":    SKILL_DISPLAY_NAMES.get(top_skill, top_skill),
        "next_modality":         "story",
        "next_difficulty":       1,
        "next_difficulty_label": "very_easy",
        "hint_intensity":        "none",
        "reward_type":           "celebration",
        "session_length_adjustment": {"remaining_seconds": 0, "action": "end_session"},
        "end_session":           True,
        "end_reason":            reason,
        "parent_tip":            offline_tips.get(reason, "Great session today!"),
        "offline_activity":      _get_offline_activity(top_skill, age_band),
        "reason":                f"Session ending: {reason}.",
        "mastery_context":       {},
    }


def _get_offline_activity(skill: str, age_band: str) -> str:
    """
    Rule-based offline activity suggestion.
    Maps skill + age_band to a practical home activity.
    No ML — pure lookup table.
    """
    activities = {
        ("phonics", "toddler"):                "Sing simple nursery rhymes together.",
        ("phonics", "preschool"):              "Point out letters on cereal boxes and signs.",
        ("phonics", "early_primary"):          "Read a short story aloud together tonight.",
        ("letter_recognition", "toddler"):     "Draw letters in sand or shaving cream.",
        ("letter_recognition", "preschool"):   "Trace letters on each other's backs — a fun guessing game.",
        ("counting", "toddler"):               "Count toys while tidying up together.",
        ("counting", "preschool"):             "Count steps on the stairs or fruit in the bowl.",
        ("counting", "early_primary"):         "Play a counting board game like Snakes and Ladders.",
        ("number_order", "preschool"):         "Arrange toys by size from smallest to biggest.",
        ("number_order", "early_primary"):     "Put fridge magnets in number order.",
        ("shape_recognition", "toddler"):      "Name shapes of everyday objects — plates, windows, doors.",
        ("color_sorting", "toddler"):          "Sort laundry or toys by color together.",
        ("hand_eye_coordination", "preschool"):"Try threading pasta onto string or playdough modelling.",
        ("attention_retention", "preschool"):  "Play a short 'I Spy' game — 5 minutes maximum.",
        ("memory_recall", "early_primary"):    "Play a simple card memory match game.",
        ("social_response", "preschool"):      "Have a short puppet show with turn-taking.",
        ("emotional_regulation", "preschool"): "Read a feelings book — name the emotion on each page.",
    }

    # Try exact match first, then skill-only fallback
    key = (skill, age_band)
    if key in activities:
        return activities[key]

    skill_fallbacks = {
        "phonics":               "Sing a favourite song together.",
        "letter_recognition":    "Spot letters on packaging around the house.",
        "counting":              "Count everyday objects together.",
        "shape_recognition":     "Find shapes in pictures in a magazine.",
        "color_sorting":         "Sort objects around the house by colour.",
        "hand_eye_coordination": "Try some simple arts and crafts.",
        "attention_retention":   "Play a 5-minute quiet game together.",
        "memory_recall":         "Play a simple memory card game.",
        "social_response":       "Do a short role-play with toys.",
        "emotional_regulation":  "Talk about feelings during the day.",
        "number_order":          "Put things in order — tallest to shortest.",
    }
    return skill_fallbacks.get(skill, "Take a fun activity break together.")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _clamp_difficulty(diff: int, age_band: str) -> int:
    from config import AGE_BAND_START_DIFFICULTY
    start  = AGE_BAND_START_DIFFICULTY.get(age_band, 2)
    lo     = max(DIFFICULTY_MIN, start - 1)
    hi     = min(DIFFICULTY_MAX, start + 2)
    return max(lo, min(hi, diff))


def _classify_mastery(prob: float) -> str:
    if prob < 0.40:
        return "low"
    elif prob < 0.70:
        return "developing"
    return "high"
