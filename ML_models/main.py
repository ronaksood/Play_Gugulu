# =============================================================================
# main.py — Adaptive Learning Intelligence Pipeline
#
# Entry point. Wires all components together.
#
# Usage
# -----
# Python API:
#   pipeline = AdaptiveLearningPipeline()
#   profile  = pipeline.onboard(onboarding_data)
#   outputs  = pipeline.process_session(session_events, child_profile, session_meta)
#
# Demo (no training required):
#   python main.py
#
# Full pipeline (requires trained models):
#   python training/train_skill_mastery.py
#   python training/train_engagement.py
#   python main.py --trained
# =============================================================================

import json
import uuid
import time
import argparse
from datetime import datetime, timezone
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import SKILLS, MODEL_DIR, AGE_SESSION_CAPS

from engine.profile_builder import build_starter_profile
from engine.skill_graph      import update_skill_graph
from engine.formula_engine   import decide_next_action
from engine.insight_generator import generate_parent_summary, generate_session_summary

from models.skill_mastery      import SkillMasteryClassifier
from models.engagement_classifier import EngagementClassifier


# ---------------------------------------------------------------------------
# Pipeline Class
# ---------------------------------------------------------------------------

class AdaptiveLearningPipeline:
    """
    End-to-end adaptive learning engine.

    Exactly 2 ML models:
        1. SkillMasteryClassifier   — XGBoost, per-skill mastery probabilities
        2. EngagementClassifier     — XGBoost, 3-class engagement state

    Everything else: deterministic formula-based logic.
    All outputs: JSON-serializable dicts.
    """

    def __init__(self, model_dir: str = MODEL_DIR):
        self.skill_model      = SkillMasteryClassifier(model_dir=model_dir)
        self.engagement_model = EngagementClassifier(
            model_path=os.path.join(model_dir, "engagement_classifier.joblib")
        )
        self._log(f"Pipeline ready. Models trained: skill={self.skill_model.is_trained()}, "
                  f"engagement={self.engagement_model.is_trained()}")

    # -----------------------------------------------------------------------
    # 1. Onboarding
    # -----------------------------------------------------------------------

    def onboard(self, onboarding_data: dict) -> dict:
        """
        Build a starter child profile from parent onboarding inputs.

        Input  : OnboardingData dict
        Output : starter_profile dict (child_profile.json)
        """
        profile = build_starter_profile(onboarding_data)
        self._log(f"Profile created for '{profile['name']}' (age_band={profile['age_band']})")
        return profile

    # -----------------------------------------------------------------------
    # 2. Full Session Processing
    # -----------------------------------------------------------------------

    def process_session(
        self,
        session_events: list[dict],
        child_profile: dict,
        session_meta: dict | None = None,
    ) -> dict:
        """
        Run the full inference pipeline for a completed (or in-progress) session.

        Parameters
        ----------
        session_events  : list of GameplayEvent dicts
        child_profile   : current ChildProfile dict (modified in place)
        session_meta    : optional dict with:
                          - session_id       : str
                          - elapsed_seconds  : float
                          - allowed_minutes  : int

        Returns
        -------
        dict with keys:
            skill_graph       : skill_graph.json output
            engagement_state  : engagement_state.json output
            next_activity     : next_activity.json output
            parent_summary    : parent_summary.json output
            session_summary   : session_summary.json output
        All values are JSON-serializable dicts.
        """
        t_start = time.perf_counter()

        if session_meta is None:
            session_meta = {}

        session_id = session_meta.get("session_id", str(uuid.uuid4()))
        child_profile["session_id"] = session_id

        elapsed_seconds  = float(session_meta.get("elapsed_seconds", len(session_events) * 45))
        allowed_seconds  = float(session_meta.get("allowed_minutes",
                                  child_profile.get("allowed_session_minutes", 15))) * 60.0

        # -------------------------------------------------------------------
        # Step 1: Skill Mastery Inference (ML Model 1)
        # -------------------------------------------------------------------
        mastery_predictions = self.skill_model.predict(session_events, child_profile)

        # -------------------------------------------------------------------
        # Step 2: Engagement / Fatigue Inference (ML Model 2)
        # -------------------------------------------------------------------
        engagement_result = self.engagement_model.predict(
            session_events, child_profile,
            elapsed_seconds, allowed_seconds
        )

        # -------------------------------------------------------------------
        # Step 3: Skill Graph Update (formula-based)
        # -------------------------------------------------------------------
        skill_graph, child_profile = update_skill_graph(
            child_profile, mastery_predictions, session_events, session_id
        )

        # -------------------------------------------------------------------
        # Step 4: Next Activity Decision (formula-based)
        # -------------------------------------------------------------------
        next_activity = decide_next_action(
            child_profile, skill_graph, engagement_result,
            session_events, elapsed_seconds
        )

        # -------------------------------------------------------------------
        # Step 5: Parent Summary (formula-based text)
        # -------------------------------------------------------------------
        parent_summary = generate_parent_summary(
            child_profile, skill_graph, engagement_result,
            session_events, elapsed_seconds
        )

        # -------------------------------------------------------------------
        # Step 6: Session Summary (formula-based)
        # -------------------------------------------------------------------
        session_summary = generate_session_summary(
            child_profile, skill_graph, engagement_result,
            session_events, elapsed_seconds
        )

        elapsed_ms = round((time.perf_counter() - t_start) * 1000, 2)

        # Package full output
        output = {
            "pipeline_meta": {
                "child_id":          child_profile.get("child_id"),
                "session_id":        session_id,
                "processed_at":      datetime.now(timezone.utc).isoformat(),
                "inference_ms":      elapsed_ms,
                "models_trained": {
                    "skill_mastery":  self.skill_model.is_trained(),
                    "engagement":     self.engagement_model.is_trained(),
                },
                "n_events":          len(session_events),
            },
            "skill_graph":      skill_graph,
            "engagement_state": engagement_result,
            "next_activity":    next_activity,
            "parent_summary":   parent_summary,
            "session_summary":  session_summary,
        }

        self._log(
            f"Session processed: {len(session_events)} events | "
            f"engagement={engagement_result['engagement_state']} | "
            f"next={next_activity['next_skill_focus']} | "
            f"{elapsed_ms}ms"
        )

        return output

    # -----------------------------------------------------------------------
    # 3. Convenience getters
    # -----------------------------------------------------------------------

    def get_engagement_state(
        self,
        session_events: list[dict],
        child_profile: dict,
        elapsed_seconds: float,
        allowed_seconds: float,
    ) -> dict:
        """Lightweight: only run Model 2."""
        return self.engagement_model.predict(
            session_events, child_profile, elapsed_seconds, allowed_seconds
        )

    def get_skill_probabilities(
        self,
        session_events: list[dict],
        child_profile: dict,
    ) -> dict:
        """Lightweight: only run Model 1."""
        return self.skill_model.predict(session_events, child_profile)

    # -----------------------------------------------------------------------
    # Internal
    # -----------------------------------------------------------------------

    @staticmethod
    def _log(msg: str):
        ts = datetime.now().strftime("%H:%M:%S")
        print(f"[{ts}] {msg}")


# ---------------------------------------------------------------------------
# Demo / CLI entry point
# ---------------------------------------------------------------------------

def run_demo(trained_models: bool = False):
    """
    End-to-end demo using synthetic data.
    Works with or without trained models (uses rule-based fallbacks).
    """
    print("\n" + "="*60)
    print("  ADAPTIVE LEARNING INTELLIGENCE — DEMO")
    print("="*60)

    from data.training_schema import SyntheticDataGenerator

    gen = SyntheticDataGenerator(seed=7)

    # 1. Parent onboarding
    print("\n[1] PARENT ONBOARDING")
    onboarding = {
        "child_name":              "Aarav",
        "child_age_years":         5.5,
        "language":                "en",
        "interests":               ["animals", "music", "stories"],
        "goals":                   ["improve counting", "learn phonics"],
        "sensitivities":           [],
        "allowed_session_minutes": 15,
    }
    print(f"    Child: {onboarding['child_name']}, Age: {onboarding['child_age_years']}y")
    print(f"    Goals: {onboarding['goals']}")

    # 2. Build starter profile
    pipeline = AdaptiveLearningPipeline()
    profile  = pipeline.onboard(onboarding)

    print(f"\n[2] STARTER PROFILE")
    print(f"    age_band         : {profile['age_band']}")
    print(f"    preferred_modality: {profile['preferred_modality']}")
    print(f"    session_cap      : {profile['allowed_session_minutes']} min")
    print(f"    unlocked_skills  : {len(profile['unlocked_skills'])} skills")
    print(f"    first_session_plan: {profile['recommended_first_session_plan']['main_skill']}")

    # 3. Simulate a gameplay session
    print(f"\n[3] SIMULATING SESSION (25 events, mixed engagement)")
    events = gen.generate_session_events(
        {**profile, "session_id": "demo_session_001"},
        n_events=25,
        engagement_scenario="mixed",
    )
    print(f"    Generated {len(events)} events across skills: "
          f"{list({e['skill'] for e in events})[:4]}...")

    # 4. Run full pipeline
    print(f"\n[4] RUNNING PIPELINE (2 ML models + formula engine)...")
    session_meta = {
        "session_id":      "demo_session_001",
        "elapsed_seconds": 720,    # 12 minutes
        "allowed_minutes": 15,
    }
    output = pipeline.process_session(events, profile, session_meta)

    # 5. Print outputs
    print(f"\n[5] ENGAGEMENT STATE")
    eng = output["engagement_state"]
    print(f"    State      : {eng['engagement_state']}")
    print(f"    Fatigue risk: {eng['fatigue_risk']}")
    print(f"    Alert level : {eng['alert_level']}")
    print(f"    Cold start  : {eng['cold_start']}")

    print(f"\n[6] SKILL GRAPH (top 5 skills)")
    nodes = sorted(output["skill_graph"]["skill_nodes"],
                   key=lambda n: n["mastery_prob"], reverse=True)
    for n in nodes[:5]:
        print(f"    {n['skill_name']:25} {n['mastery_class']:12} "
              f"p={n['mastery_prob']:.3f}  trend={n['trend']}")

    print(f"\n[7] NEXT ACTIVITY")
    na = output["next_activity"]
    print(f"    Skill      : {na['next_skill_focus']}")
    print(f"    Difficulty : {na['next_difficulty_label']}")
    print(f"    Modality   : {na['next_modality']}")
    print(f"    Hints      : {na['hint_intensity']}")
    print(f"    Reward     : {na['reward_type']}")
    print(f"    Reason     : {na['reason']}")

    print(f"\n[8] PARENT SUMMARY")
    ps = output["parent_summary"]
    print(f"    Summary    : {ps['human_summary']}")
    if ps["strengths"]:
        print(f"    Strengths  : {[s['display_name'] for s in ps['strengths'][:2]]}")
    if ps["weak_areas"]:
        print(f"    Needs work : {[s['display_name'] for s in ps['weak_areas'][:2]]}")
    print(f"    Parent tip : {na.get('parent_tip', '')}")
    print(f"    Offline    : {ps['recommended_offline_activity']}")

    print(f"\n[9] SESSION SUMMARY")
    ss = output["session_summary"]
    print(f"    Duration   : {ss['duration_minutes']} min")
    print(f"    Tasks done : {ss['tasks_completed']}")
    print(f"    Accuracy   : {int(ss['overall_accuracy']*100)}%")
    print(f"    Summary    : {ss['summary_for_parent']}")

    print(f"\n[10] INFERENCE TIME: {output['pipeline_meta']['inference_ms']} ms")

    # Save full output to file
    out_path = "examples/demo_full_output.json"
    os.makedirs("examples", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nFull JSON output saved -> {out_path}")
    print("="*60 + "\n")

    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Adaptive Learning Pipeline")
    parser.add_argument("--trained", action="store_true",
                        help="Use trained models (run training scripts first)")
    args = parser.parse_args()

    # Check if models exist
    any_model_exists = any(
        os.path.exists(os.path.join(MODEL_DIR, f"{s}_mastery.joblib"))
        for s in SKILLS
    )
    if args.trained and not any_model_exists:
        print("ERROR: No trained models found.")
        print("Run these first:")
        print("  python training/train_skill_mastery.py")
        print("  python training/train_engagement.py")
        sys.exit(1)

    run_demo(trained_models=args.trained)
