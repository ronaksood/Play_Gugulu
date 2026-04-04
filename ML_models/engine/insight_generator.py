# =============================================================================
# engine/insight_generator.py
# Rule-based parent insight and session summary generator.
# NO ML. Converts model outputs + skill graph into human-friendly JSON.
# All text is simple, positive, and parent-appropriate.
# =============================================================================

from datetime import datetime, timezone
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    SKILLS,
    SKILL_DISPLAY_NAMES,
    MASTERY_LOW_THRESHOLD,
    MASTERY_HIGH_THRESHOLD,
)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_parent_summary(
    child_profile: dict,
    skill_graph: dict,
    engagement_result: dict,
    session_events: list[dict],
    session_elapsed_seconds: float,
) -> dict:
    """
    Generate parent_summary.json

    Returns
    -------
    parent_summary dict (matches schemas/parent_summary_schema.json)
    """
    child_id      = child_profile.get("child_id", "unknown")
    session_id    = child_profile.get("session_id", "unknown")
    name          = child_profile.get("name", "Your child")
    age_band      = child_profile.get("age_band", "preschool")
    mastery_probs = child_profile.get("skill_mastery_probs", {})
    skill_nodes   = {n["skill_name"]: n for n in skill_graph.get("skill_nodes", [])}

    engagement_state = engagement_result.get("engagement_state", "engaged")
    fatigue_risk     = engagement_result.get("fatigue_risk", 0.0)

    strengths       = _identify_strengths(mastery_probs, skill_nodes)
    weak_areas      = _identify_weak_areas(mastery_probs, skill_nodes)
    best_style      = _detect_learning_style(child_profile, session_events)
    attention_trend = _describe_attention_trend(engagement_result, session_elapsed_seconds)
    offline_activity = _pick_offline_activity(mastery_probs, age_band)
    parent_action   = _generate_parent_action(weak_areas, engagement_state, age_band, name)
    confidence      = _compute_summary_confidence(skill_nodes, len(session_events))
    human_summary   = _write_human_summary(
        name, strengths, weak_areas, engagement_state,
        session_elapsed_seconds, best_style
    )

    return {
        "child_id":                  child_id,
        "session_id":                session_id,
        "generated_at":              datetime.now(timezone.utc).isoformat(),
        "strengths":                 strengths,
        "weak_areas":                weak_areas,
        "best_learning_style":       best_style,
        "attention_trend":           attention_trend,
        "recommended_offline_activity": offline_activity,
        "parent_action":             parent_action,
        "confidence":                confidence,
        "human_summary":             human_summary,
    }


def generate_session_summary(
    child_profile: dict,
    skill_graph: dict,
    engagement_result: dict,
    session_events: list[dict],
    session_elapsed_seconds: float,
) -> dict:
    """
    Generate session_summary.json

    Returns
    -------
    session_summary dict (matches schemas/session_summary_schema.json)
    """
    child_id   = child_profile.get("child_id", "unknown")
    session_id = child_profile.get("session_id", "unknown")
    name       = child_profile.get("name", "Your child")
    age_band   = child_profile.get("age_band", "preschool")

    tasks_completed  = len(session_events)
    skills_seen      = list({e.get("skill") for e in session_events if e.get("skill")})
    skill_nodes      = {n["skill_name"]: n for n in skill_graph.get("skill_nodes", [])}
    mastery_probs    = child_profile.get("skill_mastery_probs", {})

    skills_improved = [
        s for s in skills_seen
        if skill_nodes.get(s, {}).get("trend") == "improving"
    ]
    skills_support  = [
        s for s in skills_seen
        if skill_nodes.get(s, {}).get("reinforcement_need") in ("urgent", "high")
    ]

    engagement_notes  = _summarise_engagement(engagement_result, session_elapsed_seconds)
    next_session_time = _recommend_next_session_time(engagement_result, age_band)
    summary_for_parent = _write_session_summary_text(
        name, tasks_completed, skills_improved, skills_support,
        engagement_result, session_elapsed_seconds
    )

    # Accuracy stats
    correct = sum(1 for e in session_events if e.get("correct", False))
    accuracy = round(correct / tasks_completed, 3) if tasks_completed > 0 else 0.0

    return {
        "child_id":                    child_id,
        "session_id":                  session_id,
        "generated_at":                datetime.now(timezone.utc).isoformat(),
        "duration_minutes":            round(session_elapsed_seconds / 60, 1),
        "tasks_completed":             tasks_completed,
        "overall_accuracy":            accuracy,
        "skills_practiced":            skills_seen,
        "skills_improved":             skills_improved,
        "skills_improved_display":     [SKILL_DISPLAY_NAMES.get(s, s) for s in skills_improved],
        "skills_needing_support":      skills_support,
        "skills_needing_support_display": [SKILL_DISPLAY_NAMES.get(s, s) for s in skills_support],
        "engagement_notes":            engagement_notes,
        "recommended_next_session_time": next_session_time,
        "summary_for_parent":          summary_for_parent,
    }


# ---------------------------------------------------------------------------
# Private sub-functions
# ---------------------------------------------------------------------------

def _identify_strengths(mastery_probs: dict, skill_nodes: dict) -> list[dict]:
    """Skills with mastery_prob >= 0.70, sorted descending."""
    strengths = []
    for skill in SKILLS:
        prob = mastery_probs.get(skill, 0.0)
        if prob >= MASTERY_HIGH_THRESHOLD:
            node = skill_nodes.get(skill, {})
            strengths.append({
                "skill":        skill,
                "display_name": SKILL_DISPLAY_NAMES.get(skill, skill),
                "mastery_prob": round(prob, 3),
                "trend":        node.get("trend", "stable"),
                "note":         _strength_note(skill, prob, node.get("trend", "stable")),
            })
    return sorted(strengths, key=lambda x: x["mastery_prob"], reverse=True)


def _identify_weak_areas(mastery_probs: dict, skill_nodes: dict) -> list[dict]:
    """Skills with mastery_prob < 0.40, sorted ascending."""
    weak = []
    for skill in SKILLS:
        prob = mastery_probs.get(skill, 0.0)
        if prob < MASTERY_LOW_THRESHOLD:
            node = skill_nodes.get(skill, {})
            weak.append({
                "skill":        skill,
                "display_name": SKILL_DISPLAY_NAMES.get(skill, skill),
                "mastery_prob": round(prob, 3),
                "trend":        node.get("trend", "stable"),
                "note":         _weak_area_note(skill, prob, node.get("trend", "stable")),
            })
    return sorted(weak, key=lambda x: x["mastery_prob"])


def _detect_learning_style(child_profile: dict, session_events: list) -> dict:
    """
    Infer best learning style from modality success rates this session.
    Falls back to preferred_modality if insufficient data.
    """
    preferred = child_profile.get("preferred_modality", "visual")
    modality_data = {}

    for m in ["visual", "auditory", "kinesthetic", "story", "mixed"]:
        events = [e for e in session_events if e.get("modality") == m]
        if len(events) >= 3:
            rate = sum(1 for e in events if e.get("correct", False)) / len(events)
            modality_data[m] = round(rate, 3)

    if not modality_data:
        return {
            "modality":   preferred,
            "confidence": "low",
            "note":       _modality_tip(preferred),
        }

    best = max(modality_data, key=lambda m: modality_data[m])
    best_rate = modality_data[best]

    label_map = {
        "visual":      "visual activities",
        "auditory":    "sound and music activities",
        "kinesthetic": "movement and hands-on activities",
        "story":       "storytelling and reading",
        "mixed":       "mixed activity formats",
    }

    return {
        "modality":         best,
        "success_rate":     best_rate,
        "confidence":       "high" if best_rate >= 0.75 else "moderate",
        "display":          label_map.get(best, best),
        "note":             _modality_tip(best),
    }


def _describe_attention_trend(engagement_result: dict, elapsed_seconds: float) -> dict:
    """Convert engagement model output into parent-friendly attention description."""
    state        = engagement_result.get("engagement_state", "engaged")
    fatigue_risk = engagement_result.get("fatigue_risk", 0.0)
    elapsed_min  = elapsed_seconds / 60.0

    if state == "engaged":
        quality = "strong"
        note    = f"Focus stayed strong throughout the {int(elapsed_min)}-minute session."
    elif state == "neutral":
        quality = "moderate"
        note    = "Focus was steady, with a few slower moments toward the end."
    else:  # fatigued
        quality = "short"
        note    = f"Attention started well but faded after about {int(elapsed_min * 0.6):.0f} minutes — that is normal!"

    return {
        "quality":      quality,
        "engagement":   state,
        "fatigue_risk": round(fatigue_risk, 3),
        "note":         note,
    }


def _pick_offline_activity(mastery_probs: dict, age_band: str) -> str:
    """Pick an offline home activity for the lowest-mastery skill."""
    if not mastery_probs:
        return "Play a simple counting game together at home."

    weakest_skill = min(mastery_probs, key=lambda s: mastery_probs.get(s, 1.0))

    from engine.formula_engine import _get_offline_activity
    return _get_offline_activity(weakest_skill, age_band)


def _generate_parent_action(
    weak_areas: list,
    engagement_state: str,
    age_band: str,
    name: str,
) -> str:
    """One clear, actionable parent instruction."""
    if engagement_state == "fatigued":
        return f"Let {name} have a rest. Short 5-minute breaks between sessions work really well."

    if not weak_areas:
        return f"{name} is doing great! Keep sessions regular and try the recommended home activity."

    top_weak = weak_areas[0]
    display  = top_weak["display_name"]
    trend    = top_weak.get("trend", "stable")

    if trend == "declining":
        return f"Focus on {display} this week — it needs a little extra attention. Little and often works best."

    age_tips = {
        "toddler":       f"Try short, playful {display} moments during everyday routines.",
        "preschool":     f"Incorporate {display} into play — games and songs work well at this age.",
        "early_primary": f"5 minutes of {display} practice each day makes a big difference.",
        "primary":       f"Encourage {name} to practice {display} independently for a few minutes each day.",
    }
    return age_tips.get(age_band, f"A little {display} practice each day will help.")


def _compute_summary_confidence(skill_nodes: dict, n_events: int) -> str:
    """Summarise overall confidence in the session's outputs."""
    if n_events < 5:
        return "low"
    cold_starts = sum(1 for n in skill_nodes.values() if n.get("cold_start", True))
    cold_ratio  = cold_starts / max(len(skill_nodes), 1)
    if cold_ratio > 0.7:
        return "low"
    elif cold_ratio > 0.3:
        return "moderate"
    return "high"


def _write_human_summary(
    name: str,
    strengths: list,
    weak_areas: list,
    engagement_state: str,
    elapsed_seconds: float,
    best_style: dict,
) -> str:
    """
    Write a single, warm, human-readable paragraph for parents.
    Rule-based string construction — no ML.
    """
    elapsed_min = int(elapsed_seconds / 60)
    parts       = []

    # Opening
    if engagement_state == "engaged":
        parts.append(f"{name} had a great {elapsed_min}-minute session today and stayed focused throughout.")
    elif engagement_state == "neutral":
        parts.append(f"{name} completed a {elapsed_min}-minute session today with good effort.")
    else:
        parts.append(f"{name} worked hard today. The session wrapped up a little early — that is completely normal.")

    # Strengths
    if strengths:
        top = strengths[0]["display_name"]
        parts.append(f"{top} is looking really strong.")
        if len(strengths) > 1:
            second = strengths[1]["display_name"]
            parts.append(f"{second} is also going well.")

    # Weak areas
    if weak_areas:
        top_weak = weak_areas[0]["display_name"]
        trend    = weak_areas[0].get("trend", "stable")
        if trend == "improving":
            parts.append(f"{top_weak} is improving — keep up the regular practice.")
        else:
            parts.append(f"{top_weak} needs a little more practice — try the home activity suggestion below.")

    # Learning style
    style_note = best_style.get("note", "")
    if style_note:
        parts.append(style_note)

    return " ".join(parts)


def _write_session_summary_text(
    name: str,
    tasks_completed: int,
    skills_improved: list,
    skills_support: list,
    engagement_result: dict,
    elapsed_seconds: float,
) -> str:
    """Plain-language session summary for parent dashboard."""
    elapsed_min = int(elapsed_seconds / 60)
    state       = engagement_result.get("engagement_state", "engaged")

    parts = [f"{name} completed {tasks_completed} activities in {elapsed_min} minutes."]

    if skills_improved:
        improved_names = [SKILL_DISPLAY_NAMES.get(s, s) for s in skills_improved[:2]]
        parts.append(f"Good progress on: {', '.join(improved_names)}.")

    if skills_support:
        support_names = [SKILL_DISPLAY_NAMES.get(s, s) for s in skills_support[:2]]
        parts.append(f"Extra practice will help with: {', '.join(support_names)}.")

    if state == "fatigued":
        parts.append("Focus dipped toward the end — a short rest before next time will help.")
    elif state == "engaged":
        parts.append("Focus and energy were great today!")

    return " ".join(parts)


def _summarise_engagement(engagement_result: dict, elapsed_seconds: float) -> dict:
    state     = engagement_result.get("engagement_state", "engaged")
    fat_risk  = engagement_result.get("fatigue_risk", 0.0)
    elapsed_m = round(elapsed_seconds / 60, 1)

    notes_map = {
        "engaged":  f"Strong focus maintained for {elapsed_m} minutes.",
        "neutral":  f"Steady engagement with some slower moments at {elapsed_m} minutes.",
        "fatigued": f"Signs of tiredness appeared around {int(elapsed_m * 0.65):.0f} minutes.",
    }
    return {
        "state":       state,
        "fatigue_risk": round(fat_risk, 3),
        "duration_min": elapsed_m,
        "note":         notes_map.get(state, ""),
    }


def _recommend_next_session_time(engagement_result: dict, age_band: str) -> str:
    state = engagement_result.get("engagement_state", "engaged")
    fat   = engagement_result.get("fatigue_risk", 0.0)

    if state == "fatigued" or fat >= 0.65:
        return "tomorrow_or_later"
    if state == "neutral":
        return "later_today_or_tomorrow"
    return "today_or_tomorrow"


# ---------------------------------------------------------------------------
# Micro text helpers
# ---------------------------------------------------------------------------

def _strength_note(skill: str, prob: float, trend: str) -> str:
    display = SKILL_DISPLAY_NAMES.get(skill, skill)
    if trend == "improving":
        return f"{display} is improving and looking strong."
    if trend == "stable" and prob >= 0.85:
        return f"{display} is well established."
    return f"{display} is a clear strength."


def _weak_area_note(skill: str, prob: float, trend: str) -> str:
    display = SKILL_DISPLAY_NAMES.get(skill, skill)
    if trend == "improving":
        return f"{display} is improving — keep going."
    if trend == "declining":
        return f"{display} needs focused practice this week."
    return f"{display} is still developing — consistent practice helps."


def _modality_tip(modality: str) -> str:
    tips = {
        "visual":      "Your child learns best through visual activities — use pictures, colours and patterns.",
        "auditory":    "Your child responds well to sounds and music — songs and rhymes work brilliantly.",
        "kinesthetic": "Your child learns best through movement — active and hands-on games are ideal.",
        "story":       "Storytelling works really well for your child — books and imaginative play are great.",
        "mixed":       "Your child adapts well to different activity types — variety keeps them engaged.",
    }
    return tips.get(modality, "Regular, varied practice sessions work well for your child.")
