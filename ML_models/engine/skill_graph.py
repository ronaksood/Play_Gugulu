# =============================================================================
# engine/skill_graph.py
# Formula-based skill graph updater.
# Converts ML mastery predictions into a live, evolving skill graph JSON.
# No ML here — pure deterministic updates.
# =============================================================================

from datetime import datetime, timezone
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    SKILLS,
    SKILL_PREREQUISITES,
    SKILL_DISPLAY_NAMES,
    MASTERY_EMA_ALPHA,
    REGRESSION_ALERT_DROP,
    PREREQUISITE_GATE_THRESHOLD,
    MASTERY_LOW_THRESHOLD,
    MASTERY_HIGH_THRESHOLD,
)


def update_skill_graph(
    child_profile: dict,
    mastery_predictions: dict,
    session_events: list[dict],
    session_id: str,
) -> dict:
    """
    Merge new ML mastery predictions into the child profile skill graph
    using Exponential Moving Average smoothing.

    Parameters
    ----------
    child_profile       : current ChildProfile dict
    mastery_predictions : output of SkillMasteryClassifier.predict()
    session_events      : gameplay events from current session
    session_id          : current session identifier

    Returns
    -------
    skill_graph dict (matches schemas/skill_graph_schema.json)
    """
    child_id    = child_profile.get("child_id", "unknown")
    old_probs   = child_profile.get("skill_mastery_probs", {})
    now         = datetime.now(timezone.utc).isoformat()

    skill_nodes = []

    for skill in SKILLS:
        pred = mastery_predictions.get(skill, {})
        new_prob    = pred.get("mastery_prob", old_probs.get(skill, 0.1))
        confidence  = pred.get("confidence",  0.40)
        cold_start  = pred.get("cold_start",  True)

        old_prob    = old_probs.get(skill, new_prob)

        # EMA smoothing: dampen single-session noise
        # When cold start or low confidence → lean more on old value
        alpha = MASTERY_EMA_ALPHA if (not cold_start and confidence >= 0.55) else 0.30
        smoothed = round(alpha * new_prob + (1.0 - alpha) * old_prob, 4)

        # Trend detection
        trend = _compute_trend(old_prob, smoothed)

        # Regression detection
        regression_flagged = (old_prob - smoothed) >= REGRESSION_ALERT_DROP

        # Reinforcement need
        reinforcement_need = _compute_reinforcement_need(smoothed, trend, regression_flagged)

        # Evidence summary (human-readable, rule-based)
        evidence = _build_evidence_summary(
            skill, smoothed, old_prob, trend,
            session_events, confidence, cold_start
        )

        # Prerequisite status
        prereqs     = SKILL_PREREQUISITES.get(skill, [])
        prereq_met  = all(
            _get_current_prob(p, skill_nodes, old_probs) >= PREREQUISITE_GATE_THRESHOLD
            for p in prereqs
        )

        node = {
            "skill_name":          skill,
            "skill_display_name":  SKILL_DISPLAY_NAMES[skill],
            "mastery_prob":        smoothed,
            "mastery_class":       _classify_mastery(smoothed),
            "previous_mastery":    round(old_prob, 4),
            "confidence":          round(confidence, 4),
            "trend":               trend,
            "regression_flagged":  regression_flagged,
            "reinforcement_need":  reinforcement_need,
            "prerequisite_skills": prereqs,
            "prerequisite_met":    prereq_met,
            "unlocked":            len(prereqs) == 0 or prereq_met,
            "evidence_summary":    evidence,
            "cold_start":          cold_start,
            "last_updated":        now,
            "session_id":          session_id,
        }
        skill_nodes.append(node)

    # Update profile in-place
    updated_probs = {node["skill_name"]: node["mastery_prob"] for node in skill_nodes}
    child_profile["skill_mastery_probs"] = updated_probs
    child_profile["unlocked_skills"]     = [
        node["skill_name"] for node in skill_nodes if node["unlocked"]
    ]
    child_profile["updated_at"] = now
    child_profile["flags"]["cold_start"]        = False
    child_profile["flags"]["needs_calibration"] = False

    skill_graph = {
        "child_id":    child_id,
        "session_id":  session_id,
        "generated_at": now,
        "skill_nodes": skill_nodes,
        "summary": {
            "skills_mastered":      [n["skill_name"] for n in skill_nodes if n["mastery_class"] == "high"],
            "skills_developing":    [n["skill_name"] for n in skill_nodes if n["mastery_class"] == "developing"],
            "skills_need_support":  [n["skill_name"] for n in skill_nodes if n["mastery_class"] == "low"],
            "skills_regressing":    [n["skill_name"] for n in skill_nodes if n["regression_flagged"]],
            "skills_unlocked":      [n["skill_name"] for n in skill_nodes if n["unlocked"]],
            "overall_progress_pct": round(
                100 * sum(n["mastery_prob"] for n in skill_nodes) / len(skill_nodes), 1
            ),
        }
    }

    return skill_graph, child_profile


def build_skill_dag() -> dict:
    """
    Return the static prerequisite adjacency dict from config.
    Called once at startup — the DAG never changes at runtime.
    """
    return dict(SKILL_PREREQUISITES)


def get_unlocked_skills(mastery_probs: dict) -> list:
    """Return list of skills whose prerequisites are satisfied."""
    unlocked = []
    for skill, prereqs in SKILL_PREREQUISITES.items():
        if all(mastery_probs.get(p, 0.0) >= PREREQUISITE_GATE_THRESHOLD for p in prereqs):
            unlocked.append(skill)
    return sorted(unlocked)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _compute_trend(old_prob: float, new_prob: float) -> str:
    delta = new_prob - old_prob
    if delta >= 0.05:
        return "improving"
    elif delta <= -0.05:
        return "declining"
    return "stable"


def _compute_reinforcement_need(
    mastery_prob: float,
    trend: str,
    regression_flagged: bool,
) -> str:
    """
    Formula:
    - LOW mastery + declining → "urgent"
    - LOW mastery + stable → "high"
    - DEVELOPING + declining or regression → "moderate"
    - HIGH mastery → "low"
    - DEVELOPING + improving → "low"
    """
    if regression_flagged:
        return "urgent"
    if mastery_prob < MASTERY_LOW_THRESHOLD:
        return "urgent" if trend == "declining" else "high"
    if mastery_prob < MASTERY_HIGH_THRESHOLD:
        if trend == "declining":
            return "moderate"
        return "low"
    return "low"  # mastered


def _build_evidence_summary(
    skill: str,
    new_prob: float,
    old_prob: float,
    trend: str,
    session_events: list,
    confidence: float,
    cold_start: bool,
) -> str:
    """
    Produce a human-readable evidence string for parents/debugging.
    Rule-based string construction — no ML.
    """
    display   = SKILL_DISPLAY_NAMES.get(skill, skill)
    skill_ev  = [e for e in session_events if e.get("skill") == skill]
    n         = len(skill_ev)

    if cold_start or n == 0:
        return f"No data yet for {display}. Estimate based on age group."

    correct = sum(1 for e in skill_ev if e.get("correct", False))
    acc     = correct / n if n > 0 else 0.0
    hints   = sum(1 for e in skill_ev if e.get("hint_used", False))

    parts = [f"{correct}/{n} tasks correct ({int(acc*100)}%)"]

    if hints > 0:
        parts.append(f"used hints {hints} time{'s' if hints > 1 else ''}")

    streak = 0
    for e in reversed(skill_ev):
        if e.get("correct"):
            streak += 1
        else:
            break
    if streak >= 3:
        parts.append(f"finished with a {streak}-task streak")

    trend_text = {"improving": "trending up", "declining": "trending down", "stable": "holding steady"}
    parts.append(trend_text.get(trend, "stable"))

    if confidence < 0.55:
        parts.append("confidence low — more data needed")

    return "; ".join(parts) + "."


def _get_current_prob(skill: str, computed_nodes: list, old_probs: dict) -> float:
    """Look up already-computed node OR fall back to old profile."""
    for node in computed_nodes:
        if node["skill_name"] == skill:
            return node["mastery_prob"]
    return old_probs.get(skill, 0.0)


def _classify_mastery(prob: float) -> str:
    if prob < 0.40:
        return "low"
    elif prob < 0.70:
        return "developing"
    return "high"
