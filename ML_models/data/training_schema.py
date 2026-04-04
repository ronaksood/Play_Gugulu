import random
import math
import uuid
from datetime import datetime, timedelta
from typing import TypedDict, Optional
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import SKILLS, MODALITIES, ENGAGEMENT_CLASSES, AGE_BANDS, AGE_BAND_PRIORS


# ---------------------------------------------------------------------------
# TypedDict schemas  (act as runtime documentation + type hints)
# ---------------------------------------------------------------------------

class GameplayEvent(TypedDict):
    event_id:                  str
    child_id:                  str
    session_id:                str
    task_id:                   str
    skill:                     str
    task_type:                 str
    modality:                  str
    difficulty:                int
    correct:                   bool
    response_time_ms:          float
    retry_count:               int
    hint_used:                 bool
    streak_length:             int
    time_on_task_ms:           float
    pause_duration_ms:         float
    inactivity_duration_ms:    float
    error_pattern:             str          # "none" | "sequence" | "random" | "consistent"
    reward_response:           float        # 0.0 – 1.0 (how enthusiastically child responded)
    movement_success:          float        # 0.0 – 1.0
    audio_success:             float        # 0.0 – 1.0
    visual_match_success:      float        # 0.0 – 1.0
    time_to_first_response_ms: float
    timestamp:                 str


class ChildProfile(TypedDict):
    child_id:                  str
    name:                      str
    age_months:                int
    age_band:                  str
    language:                  str
    interests:                 list
    goals:                     list
    sensitivities:             list
    sessions_completed:        int
    preferred_modality:        str
    current_difficulty_per_skill: dict     # skill -> int
    skill_mastery_probs:       dict        # skill -> float
    days_since_last_session:   int
    flags:                     dict
    created_at:                str
    updated_at:                str


class OnboardingData(TypedDict):
    child_name:        str
    child_age_years:   float
    language:          str
    interests:         list
    goals:             list
    sensitivities:     list
    allowed_session_minutes: int


# ---------------------------------------------------------------------------
# SyntheticDataGenerator
# ---------------------------------------------------------------------------

class SyntheticDataGenerator:
    """
    Generates synthetic training datasets for both ML models.
    Uses developmental psychology priors to make outputs believable.

    Usage
    -----
    gen = SyntheticDataGenerator(seed=42)
    X_skill, y_skill = gen.generate_skill_mastery_dataset(n_samples=5000)
    X_eng, y_eng     = gen.generate_engagement_dataset(n_samples=5000)
    events           = gen.generate_session_events(profile, n_events=30)
    """

    def __init__(self, seed: int = 42):
        random.seed(seed)
        self._rng = random.Random(seed)

    # -----------------------------------------------------------------------
    # Skill Mastery Dataset
    # -----------------------------------------------------------------------

    def generate_skill_mastery_dataset(self, n_samples: int = 5000):
        """
        Returns (features_df, labels_df) as pandas DataFrames.

        Label definition:
        - mastery = 1 if weighted_accuracy_last_10 > 0.75 AND accuracy_last_5 > 0.70
        - Labels are soft-suppressed for children with < 5 attempts (cold start)
        - Exponential recency weighting applied to accuracy
        """
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("pandas required for dataset generation: pip install pandas")

        rows_features = []
        rows_labels   = []

        for _ in range(n_samples):
            age_band  = self._rng.choice(list(AGE_BANDS.keys()))
            age_years = self._rng.uniform(*AGE_BANDS[age_band])
            age_months = int(age_years * 12)
            sessions  = self._rng.randint(0, 50)
            skill     = self._rng.choice(SKILLS)

            # True underlying mastery level (latent variable)
            base_prior   = AGE_BAND_PRIORS[age_band][skill]
            true_mastery = self._clamp(base_prior + self._rng.gauss(0.05 * sessions, 0.10), 0.02, 0.98)

            n_attempts = max(0, int(self._rng.gauss(sessions * 3, 5)))

            feats = self._make_skill_features(
                true_mastery=true_mastery,
                n_attempts=n_attempts,
                age_months=age_months,
                sessions=sessions,
                skill=skill,
                age_band=age_band,
            )

            # Label: mastery achieved if accuracy is consistently above threshold
            if n_attempts < 5:
                label = -1   # too sparse — excluded from training
            else:
                label = int(feats["accuracy_last_10"] > 0.75 and feats["accuracy_last_5"] > 0.70)

            rows_features.append(feats)
            rows_labels.append({"skill": skill, "mastery_label": label})

        features_df = pd.DataFrame(rows_features)
        labels_df   = pd.DataFrame(rows_labels)

        # Drop cold-start rows (label == -1) for training but keep for inspection
        valid_mask   = labels_df["mastery_label"] != -1
        return features_df[valid_mask].reset_index(drop=True), labels_df[valid_mask].reset_index(drop=True)

    def _make_skill_features(
        self,
        true_mastery: float,
        n_attempts: int,
        age_months: int,
        sessions: int,
        skill: str,
        age_band: str,
    ) -> dict:
        """Simulate realistic feature values from a latent mastery level."""
        noise    = lambda scale=0.08: self._rng.gauss(0, scale)
        modality = self._rng.choice(MODALITIES)

        def acc(true_m, n):
            if n == 0:
                return 0.5
            return self._clamp(true_m + noise(0.10), 0.0, 1.0)

        a5  = acc(true_mastery, min(n_attempts, 5))
        a10 = acc(true_mastery, min(n_attempts, 10))
        a_all = acc(true_mastery, n_attempts)

        rt_base  = max(500, 8000 - (true_mastery * 5000) + noise(500) * 1000)
        rt_trend = -true_mastery * 10 + noise(5)

        return {
            "accuracy_last_5":              a5,
            "accuracy_last_10":             a10,
            "accuracy_all_time":            a_all,
            "accuracy_trend":               self._clamp(true_mastery * 0.02 + noise(0.01), -0.1, 0.1),
            "response_time_mean":           max(300.0, rt_base),
            "response_time_last_3":         max(300.0, rt_base + noise(200) * 1000),
            "response_time_trend":          rt_trend,
            "response_time_cv":             self._clamp(0.4 - true_mastery * 0.2 + noise(0.05), 0.0, 1.0),
            "retry_count_mean":             max(0.0, 2.0 - true_mastery * 2.0 + noise(0.3)),
            "hint_usage_rate":              self._clamp(0.7 - true_mastery + noise(0.1), 0.0, 1.0),
            "hint_dependency_trend":        self._clamp(-true_mastery * 0.02 + noise(0.01), -0.1, 0.1),
            "current_streak":               max(0, int(true_mastery * 8 + noise(1) * 3)),
            "max_streak":                   max(0, int(true_mastery * 15 + noise(2) * 5)),
            "streak_breaks":                max(0, int((1 - true_mastery) * 5 + noise(0.5) * 3)),
            "error_burst_count":            max(0, int((1 - true_mastery) * 3 + noise(0.3) * 2)),
            "consec_errors_now":            max(0, int((1 - true_mastery) * 2 + noise(0.2))),
            "visual_success_rate":          self._clamp(true_mastery + noise(0.10), 0.0, 1.0),
            "auditory_success_rate":        self._clamp(true_mastery + noise(0.12), 0.0, 1.0),
            "kinesthetic_success_rate":     self._clamp(true_mastery + noise(0.11), 0.0, 1.0),
            "attempts_total":               n_attempts,
            "attempts_recent":              min(n_attempts, int(n_attempts * 0.3 + 2)),
            "has_min_attempts":             float(n_attempts >= 5),
            "prior_mastery_prob":           AGE_BAND_PRIORS[age_band][skill],
            "age_months":                   age_months,
            "age_band_encoded":             list(AGE_BANDS.keys()).index(age_band),
            "sessions_completed":           sessions,
            "preferred_modality_encoded":   MODALITIES.index(modality),
            "days_since_last_session":      self._rng.randint(0, 14),
        }

    # -----------------------------------------------------------------------
    # Engagement Dataset
    # -----------------------------------------------------------------------

    def generate_engagement_dataset(self, n_samples: int = 5000):
        """
        Returns (features_df, labels_series) as pandas objects.

        Label definition (weak supervision rules):
        - 2 (engaged):  RT low + RT trend negative + error_rate low + no long pauses
        - 1 (neutral):  medium signals
        - 0 (fatigued): RT high or RT trend positive or many errors or long pauses
        """
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("pandas required: pip install pandas")

        rows    = []
        labels  = []

        for _ in range(n_samples):
            true_state = self._rng.choice(ENGAGEMENT_CLASSES)  # "engaged", "neutral", "fatigued"
            feats, label = self._make_engagement_features(true_state)
            rows.append(feats)
            labels.append(label)

        return pd.DataFrame(rows), pd.Series(labels, name="engagement_label")

    def _make_engagement_features(self, true_state: str) -> tuple[dict, int]:
        n  = lambda scale=0.1: self._rng.gauss(0, scale)
        clamp = self._clamp

        if true_state == "engaged":
            rt_mean  = self._rng.uniform(600, 2000)
            rt_trend = self._rng.uniform(-20, 5)
            err_rate = self._rng.uniform(0.00, 0.20)
            pause    = self._rng.uniform(100, 2000)
            hint_r   = self._rng.uniform(0.00, 0.20)
            label    = 2
        elif true_state == "neutral":
            rt_mean  = self._rng.uniform(1800, 3500)
            rt_trend = self._rng.uniform(-5, 25)
            err_rate = self._rng.uniform(0.15, 0.40)
            pause    = self._rng.uniform(1500, 5000)
            hint_r   = self._rng.uniform(0.15, 0.40)
            label    = 1
        else:  # fatigued
            rt_mean  = self._rng.uniform(3000, 8000)
            rt_trend = self._rng.uniform(20, 100)
            err_rate = self._rng.uniform(0.35, 0.80)
            pause    = self._rng.uniform(4000, 15000)
            hint_r   = self._rng.uniform(0.35, 0.80)
            label    = 0

        elapsed  = self._rng.uniform(60, 1500)
        allowed  = self._rng.uniform(600, 1500)

        return {
            "response_time_mean":              max(200.0, rt_mean + n(200) * 1000),
            "response_time_last3_mean":        max(200.0, rt_mean * (1 + n(0.15))),
            "response_time_trend":             rt_trend + n(5),
            "response_time_zscore":            clamp(rt_trend / 20 + n(0.3), -3, 3),
            "response_time_cv":                clamp(0.2 + err_rate * 0.5 + n(0.05), 0, 1),
            "error_rate_last_5":               clamp(err_rate + n(0.05), 0, 1),
            "error_rate_last_10":              clamp(err_rate + n(0.05), 0, 1),
            "retry_rate_session":              clamp(err_rate * 0.7 + n(0.05), 0, 1),
            "consec_errors_now":               max(0, int(err_rate * 5 + n(0.3) * 2)),
            "error_burst_flag":                float(err_rate > 0.40 and self._rng.random() > 0.5),
            "pause_duration_mean":             max(0.0, pause + n(500) * 1000),
            "pause_duration_max":              max(0.0, pause * 2.5 + n(1000) * 1000),
            "inactivity_total_ms":             max(0.0, pause * elapsed / 30 + n(2000) * 1000),
            "inactivity_event_count":          max(0, int(err_rate * 4 + n(0.2) * 2)),
            "hint_usage_rate_session":         clamp(hint_r + n(0.05), 0, 1),
            "hint_rate_trend":                 clamp(hint_r * 0.05 + n(0.01), -0.1, 0.1),
            "streak_break_count":              max(0, int(err_rate * 6 + n(0.3) * 3)),
            "current_streak":                  max(0, int((1 - err_rate) * 5 + n(0.3) * 2)),
            "session_elapsed_seconds":         elapsed,
            "session_duration_ratio":          clamp(elapsed / max(allowed, 1), 0, 1),
            "tasks_per_minute":                clamp(3.0 - err_rate * 2 + n(0.3), 0.1, 10),
            "task_switch_count":               max(0, int(3 - err_rate * 2 + n(0.2) * 2)),
            "reward_response_mean":            clamp(1 - err_rate * 0.8 + n(0.1), 0, 1),
            "time_to_first_response_mean":     max(200.0, rt_mean * 0.5 + n(200) * 1000),
            "time_to_first_response_trend":    rt_trend * 0.5 + n(3),
            "age_months":                      self._rng.randint(30, 120),
            "age_band_encoded":                self._rng.randint(0, 3),
            "sessions_completed":              self._rng.randint(0, 50),
        }, label

    # -----------------------------------------------------------------------
    # Session Event Stream Generator (for demo / main.py)
    # -----------------------------------------------------------------------

    def generate_session_events(
        self,
        child_profile: dict,
        n_events: int = 20,
        engagement_scenario: str = "mixed",
    ) -> list[dict]:
        """
        Generate a realistic sequence of gameplay events for a child profile.

        engagement_scenario: "engaged" | "fatigued" | "mixed"
        """
        events    = []
        skill_rota = (SKILLS * 5)[:n_events]
        random.shuffle(skill_rota)

        base_mastery = child_profile.get("skill_mastery_probs", {})
        modality_pref = child_profile.get("preferred_modality", "visual")
        start_time = datetime.utcnow()

        for i, skill in enumerate(skill_rota):
            m        = base_mastery.get(skill, 0.3)
            progress = i / max(n_events - 1, 1)    # 0→1 over session

            # Fatigue effect over session
            if engagement_scenario == "fatigued":
                fatigue = 0.5 + progress * 0.5
            elif engagement_scenario == "engaged":
                fatigue = 0.0
            else:
                fatigue = max(0.0, progress - 0.4) * 1.2  # mixed: tires toward end

            correct_prob = self._clamp(m - fatigue * 0.3 + self._rng.gauss(0, 0.08), 0.05, 0.98)
            rt_base      = max(400, 5000 - m * 3000 + fatigue * 2000)

            event = GameplayEvent(
                event_id                  = str(uuid.uuid4()),
                child_id                  = child_profile.get("child_id", "demo_child"),
                session_id                = child_profile.get("session_id", "demo_session"),
                task_id                   = f"task_{skill}_{i:03d}",
                skill                     = skill,
                task_type                 = self._rng.choice(["match", "sort", "trace", "listen", "move"]),
                modality                  = modality_pref if self._rng.random() > 0.3 else self._rng.choice(MODALITIES),
                difficulty                = child_profile.get("current_difficulty_per_skill", {}).get(skill, 2),
                correct                   = self._rng.random() < correct_prob,
                response_time_ms          = max(200.0, rt_base + self._rng.gauss(0, rt_base * 0.25)),
                retry_count               = max(0, int(self._rng.expovariate(1 / max(0.5, 1 - m + fatigue * 0.5)))),
                hint_used                 = self._rng.random() < (0.6 - m + fatigue * 0.3),
                streak_length             = 0,   # computed downstream
                time_on_task_ms           = max(500.0, rt_base * 1.5 + self._rng.gauss(0, 1000)),
                pause_duration_ms         = max(0.0, self._rng.expovariate(1 / (500 + fatigue * 3000))),
                inactivity_duration_ms    = max(0.0, self._rng.expovariate(1 / (200 + fatigue * 5000))),
                error_pattern             = "consistent" if fatigue > 0.5 else "none",
                reward_response           = self._clamp(0.9 - fatigue * 0.7 + self._rng.gauss(0, 0.1), 0, 1),
                movement_success          = self._clamp(m - fatigue * 0.2 + self._rng.gauss(0, 0.1), 0, 1),
                audio_success             = self._clamp(m - fatigue * 0.15 + self._rng.gauss(0, 0.1), 0, 1),
                visual_match_success      = self._clamp(m - fatigue * 0.1 + self._rng.gauss(0, 0.1), 0, 1),
                time_to_first_response_ms = max(150.0, rt_base * 0.4 + self._rng.gauss(0, rt_base * 0.2)),
                timestamp                 = (start_time + timedelta(seconds=i * 45 + self._rng.uniform(10, 60))).isoformat() + "Z",
            )
            events.append(dict(event))

        # Backfill streak_length
        streak = 0
        for e in events:
            if e["correct"]:
                streak += 1
            else:
                streak = 0
            e["streak_length"] = streak

        return events

    # -----------------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------------

    @staticmethod
    def _clamp(value: float, lo: float, hi: float) -> float:
        return max(lo, min(hi, value))


# ---------------------------------------------------------------------------
# Standalone sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    gen = SyntheticDataGenerator(seed=42)

    print("Generating skill mastery dataset...")
    X_skill, y_skill = gen.generate_skill_mastery_dataset(n_samples=200)
    print(f"  Features shape: {X_skill.shape}")
    print(f"  Label distribution:\n{y_skill['mastery_label'].value_counts().to_string()}")

    print("\nGenerating engagement dataset...")
    X_eng, y_eng = gen.generate_engagement_dataset(n_samples=200)
    print(f"  Features shape: {X_eng.shape}")
    print(f"  Label distribution:\n{y_eng.value_counts().to_string()}")

    print("\nGenerating session events...")
    dummy_profile = {
        "child_id": "test_001",
        "session_id": "sess_001",
        "age_months": 60,
        "preferred_modality": "visual",
        "skill_mastery_probs": {s: 0.4 for s in SKILLS},
        "current_difficulty_per_skill": {s: 2 for s in SKILLS},
    }
    events = gen.generate_session_events(dummy_profile, n_events=10, engagement_scenario="mixed")
    print(f"  Generated {len(events)} events")
    print(f"  Sample event keys: {list(events[0].keys())}")
    print("\n✓ training_schema.py OK")
