# =============================================================================
# config.py — Central constants for the Adaptive Learning Intelligence System
# No classes, no imports, no side effects. Pure data.
# =============================================================================

# ---------------------------------------------------------------------------
# Skill Taxonomy
# ---------------------------------------------------------------------------
SKILLS = [
    "phonics",
    "letter_recognition",
    "counting",
    "number_order",
    "shape_recognition",
    "color_sorting",
    "hand_eye_coordination",
    "attention_retention",
    "memory_recall",
    "social_response",
    "emotional_regulation",
]

# Prerequisite DAG: skill -> list of skills that must be partially mastered first
# A prerequisite is "met" when its mastery_prob >= PREREQUISITE_GATE_THRESHOLD
SKILL_PREREQUISITES = {
    "phonics":               ["letter_recognition"],
    "letter_recognition":    [],
    "counting":              [],
    "number_order":          ["counting"],
    "shape_recognition":     [],
    "color_sorting":         ["shape_recognition"],
    "hand_eye_coordination": [],
    "attention_retention":   [],
    "memory_recall":         ["attention_retention"],
    "social_response":       [],
    "emotional_regulation":  ["social_response"],
}

# Skill categories — used by formula engine for modality mapping
SKILL_CATEGORIES = {
    "literacy":   ["phonics", "letter_recognition"],
    "numeracy":   ["counting", "number_order"],
    "visual":     ["shape_recognition", "color_sorting"],
    "motor":      ["hand_eye_coordination"],
    "cognitive":  ["attention_retention", "memory_recall"],
    "social":     ["social_response", "emotional_regulation"],
}

# ---------------------------------------------------------------------------
# Difficulty Levels
# ---------------------------------------------------------------------------
DIFFICULTY_LEVELS     = [1, 2, 3, 4, 5]
DIFFICULTY_MIN        = 1
DIFFICULTY_MAX        = 5
DIFFICULTY_LABELS     = {1: "very_easy", 2: "easy", 3: "medium", 4: "hard", 5: "very_hard"}

# ---------------------------------------------------------------------------
# Modalities
# ---------------------------------------------------------------------------
MODALITIES           = ["visual", "auditory", "kinesthetic", "story", "mixed"]
MODALITY_ENCODING    = {m: i for i, m in enumerate(MODALITIES)}
FATIGUE_MODALITY     = "kinesthetic"   # switch to this when fatigued
DEFAULT_MODALITY     = "visual"

# ---------------------------------------------------------------------------
# Engagement Classes
# ---------------------------------------------------------------------------
ENGAGEMENT_CLASSES       = ["fatigued", "neutral", "engaged"]   # index 0, 1, 2
ENGAGEMENT_CLASS_LABELS  = {0: "fatigued", 1: "neutral", 2: "engaged"}
ENGAGEMENT_LABEL_TO_IDX  = {v: k for k, v in ENGAGEMENT_CLASS_LABELS.items()}

# ---------------------------------------------------------------------------
# Age Bands  (inclusive, in years)
# ---------------------------------------------------------------------------
AGE_BANDS = {
    "toddler":       (2, 4),
    "preschool":     (4, 6),
    "early_primary": (6, 8),
    "primary":       (8, 11),
}

# Age-based starting mastery priors (cold-start values)
# Based on developmental milestone references — conservative estimates
AGE_BAND_PRIORS = {
    "toddler": {
        "phonics": 0.05,
        "letter_recognition": 0.10,
        "counting": 0.20,
        "number_order": 0.08,
        "shape_recognition": 0.30,
        "color_sorting": 0.35,
        "hand_eye_coordination": 0.25,
        "attention_retention": 0.18,
        "memory_recall": 0.15,
        "social_response": 0.30,
        "emotional_regulation": 0.20,
    },
    "preschool": {
        "phonics": 0.25,
        "letter_recognition": 0.35,
        "counting": 0.50,
        "number_order": 0.30,
        "shape_recognition": 0.55,
        "color_sorting": 0.60,
        "hand_eye_coordination": 0.45,
        "attention_retention": 0.35,
        "memory_recall": 0.30,
        "social_response": 0.50,
        "emotional_regulation": 0.38,
    },
    "early_primary": {
        "phonics": 0.55,
        "letter_recognition": 0.65,
        "counting": 0.70,
        "number_order": 0.60,
        "shape_recognition": 0.72,
        "color_sorting": 0.78,
        "hand_eye_coordination": 0.60,
        "attention_retention": 0.55,
        "memory_recall": 0.52,
        "social_response": 0.65,
        "emotional_regulation": 0.55,
    },
    "primary": {
        "phonics": 0.72,
        "letter_recognition": 0.80,
        "counting": 0.82,
        "number_order": 0.75,
        "shape_recognition": 0.82,
        "color_sorting": 0.85,
        "hand_eye_coordination": 0.70,
        "attention_retention": 0.68,
        "memory_recall": 0.65,
        "social_response": 0.72,
        "emotional_regulation": 0.65,
    },
}

# Default starting difficulty per age band
AGE_BAND_START_DIFFICULTY = {
    "toddler":       1,
    "preschool":     2,
    "early_primary": 3,
    "primary":       3,
}

# ---------------------------------------------------------------------------
# Mastery & Decision Thresholds
# ---------------------------------------------------------------------------
MASTERY_LOW_THRESHOLD      = 0.40   # below this => reduce difficulty
MASTERY_MID_THRESHOLD      = 0.70   # between LOW and HIGH => maintain
MASTERY_HIGH_THRESHOLD     = 0.70   # above this => increase difficulty
PREREQUISITE_GATE_THRESHOLD = 0.45  # prereq must reach this before unlocking dependent

MASTERY_EMA_ALPHA          = 0.70   # weight for new ML estimate vs old profile value
REGRESSION_ALERT_DROP      = 0.15   # mastery drop triggering regression flag

# ---------------------------------------------------------------------------
# Engagement / Fatigue Thresholds
# ---------------------------------------------------------------------------
FATIGUE_CONFIDENCE_HARD_STOP   = 0.80   # end session immediately
FATIGUE_CONFIDENCE_SOFT_ALERT  = 0.60   # shorten tasks, switch modality
HINT_ERROR_BURST_TRIGGER       = 3      # consecutive errors before hint is forced
HINT_COOLDOWN_EVENTS           = 4      # events to wait before offering another hint

# ---------------------------------------------------------------------------
# Session Parameters
# ---------------------------------------------------------------------------
SESSION_WINDOW_SIZE          = 20    # recent events used for feature extraction
MAX_SESSION_MINUTES_DEFAULT  = 20
MIN_SESSION_MINUTES          = 5
SESSION_NEAR_END_RATIO       = 0.85  # % of allowed time => trigger victory moment

# Age-based recommended session caps (minutes)
AGE_SESSION_CAPS = {
    "toddler":       10,
    "preschool":     15,
    "early_primary": 20,
    "primary":       25,
}

# ---------------------------------------------------------------------------
# Feature Engineering Constants
# ---------------------------------------------------------------------------
RESPONSE_TIME_WINDOW        = 5     # events for rolling response time features
ACCURACY_WINDOW_SHORT       = 5     # short accuracy window
ACCURACY_WINDOW_LONG        = 10    # long accuracy window
SLOW_RESPONSE_MULTIPLIER    = 2.0   # response time > median * this => slow

# ---------------------------------------------------------------------------
# Model Artifact Paths
# ---------------------------------------------------------------------------
MODEL_DIR            = "models/artifacts"
ENGAGEMENT_MODEL_PATH = f"{MODEL_DIR}/engagement_classifier.joblib"

def skill_model_path(skill: str) -> str:
    return f"{MODEL_DIR}/{skill}_mastery.joblib"

# ---------------------------------------------------------------------------
# Interests → Preferred Modality Mapping
# ---------------------------------------------------------------------------
INTEREST_TO_MODALITY = {
    "animals":     "visual",
    "music":       "auditory",
    "sports":      "kinesthetic",
    "art":         "visual",
    "stories":     "story",
    "dancing":     "kinesthetic",
    "puzzles":     "visual",
    "cars":        "visual",
    "nature":      "visual",
    "cooking":     "kinesthetic",
    "singing":     "auditory",
    "building":    "kinesthetic",
    "drawing":     "visual",
    "reading":     "story",
    "numbers":     "visual",
}

# ---------------------------------------------------------------------------
# Reward Types (deterministic selection)
# ---------------------------------------------------------------------------
REWARD_TYPES = ["sparkle", "progress_badge", "celebration", "encouragement_star", "streak_flame"]

# ---------------------------------------------------------------------------
# Skill → Natural Language Labels  (for parent-facing output)
# ---------------------------------------------------------------------------
SKILL_DISPLAY_NAMES = {
    "phonics":               "Phonics & Sounds",
    "letter_recognition":    "Letter Recognition",
    "counting":              "Counting",
    "number_order":          "Number Order",
    "shape_recognition":     "Shape Recognition",
    "color_sorting":         "Color Sorting",
    "hand_eye_coordination": "Hand-Eye Coordination",
    "attention_retention":   "Focus & Attention",
    "memory_recall":         "Memory",
    "social_response":       "Social Awareness",
    "emotional_regulation":  "Emotional Understanding",
}
