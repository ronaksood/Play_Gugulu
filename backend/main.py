"""
PlaySpark Adaptive Learning API  —  FastAPI Backend
====================================================
Run with:
    cd play-guglo/backend
    uvicorn main:app --reload --port 8000

The Vite dev server (port 3000) proxies /api/* requests here,
so the browser never needs a direct cross-origin call.
"""

import sys
import os
import uuid

# ---------------------------------------------------------------------------
# Path bootstrap — must happen before any ML_models import
# ---------------------------------------------------------------------------
_BACKEND_DIR   = os.path.dirname(os.path.abspath(__file__))
_ML_MODELS_DIR = os.path.normpath(os.path.join(_BACKEND_DIR, "..", "ML_models"))
_ARTIFACTS_DIR = os.path.join(_ML_MODELS_DIR, "models", "artifacts")

# Insert so that `from config import ...`, `from engine import ...`, etc. resolve
sys.path.insert(0, _ML_MODELS_DIR)

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List

# Import after path setup — this is ML_models/main.py
from main import AdaptiveLearningPipeline  # type: ignore

# ---------------------------------------------------------------------------
# App initialisation
# ---------------------------------------------------------------------------
app = FastAPI(
    title="PlaySpark Adaptive Learning API",
    version="1.0.0",
    description=(
        "Bridges the React frontend with the XGBoost-based adaptive "
        "learning pipeline. Two ML models: SkillMasteryClassifier + "
        "EngagementClassifier. Everything else is deterministic."
    ),
)

# CORS — allow both Vite ports in case the proxy is bypassed during debugging
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:5173",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Load pipeline once at startup — joblib models are cached in memory
# ---------------------------------------------------------------------------
pipeline = AdaptiveLearningPipeline(model_dir=_ARTIFACTS_DIR)

# ---------------------------------------------------------------------------
# Goal-ID → natural-language mapping
# The frontend stores short IDs ('phonics', 'counting', …).
# profile_builder.py reads natural-language goal text to detect keywords.
# ---------------------------------------------------------------------------
_GOAL_TEXT: dict[str, str] = {
    "phonics":  "learn phonics",
    "counting": "improve counting",
    "colors":   "learn shapes and colors",
    "nature":   "explore animals and nature",
}

# ---------------------------------------------------------------------------
# Pydantic request schemas
# ---------------------------------------------------------------------------

class OnboardingRequest(BaseModel):
    """
    Mirrors the data collected by the frontend OnboardingRoute.

    Fields
    ------
    parent_email            : validated email from step 1
    child_age               : integer 3-7 from step 2 (age buttons)
    interests               : list of IDs from step 3  ['music','stories','puzzles','art']
    goals                   : list of IDs from step 4  ['phonics','counting','colors','nature']
    child_name              : optional; frontend does not collect a name, defaults to "Learner"
    language                : ISO-639 language code; defaults to "en"
    sensitivities           : optional accessibility flags (not collected by frontend yet)
    allowed_session_minutes : parent-set cap; defaults to 15
    """
    parent_email: str
    child_age: int
    interests: List[str] = []
    goals: List[str] = []
    child_name: Optional[str] = "Learner"
    language: Optional[str] = "en"
    sensitivities: Optional[List[str]] = []
    allowed_session_minutes: Optional[int] = 15


class SessionMeta(BaseModel):
    session_id: Optional[str] = None
    elapsed_seconds: Optional[float] = 0.0
    allowed_minutes: Optional[int] = 15


class SessionProcessRequest(BaseModel):
    """
    Full session payload: gameplay events + current child profile + metadata.
    session_events is a list of GameplayEvent dicts (see ML_models schema).
    child_profile is the ChildProfile dict returned from /api/onboard.
    """
    session_events: List[dict]
    child_profile: dict
    session_meta: Optional[SessionMeta] = None


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/api/health")
def health_check():
    """
    Returns backend status and whether ML models have been trained.
    Trained models live in ML_models/models/artifacts/*.joblib.
    If not trained, the pipeline falls back to age-band priors (cold-start).
    """
    return {
        "status": "ok",
        "ml_models_dir": _ARTIFACTS_DIR,
        "models_trained": {
            "skill_mastery": pipeline.skill_model.is_trained(),
            "engagement":    pipeline.engagement_model.is_trained(),
        },
    }


@app.post("/api/onboard")
def onboard_child(request: OnboardingRequest):
    """
    Accept frontend onboarding form data and build a ChildProfile.

    Transformation performed here:
      child_age  (int 3-7)   → child_age_years (float)
      goals      (id[])      → goals (natural-language list)
      interests              → passed through; config.py maps them to modality

    Returns
    -------
    { "success": true, "child_profile": <ChildProfile JSON> }
    """
    try:
        ml_goals = [_GOAL_TEXT.get(g, g) for g in (request.goals or [])]

        ml_onboarding = {
            "child_name":              request.child_name or "Learner",
            "child_age_years":         float(request.child_age),
            "language":                request.language or "en",
            "interests":               request.interests or [],
            "goals":                   ml_goals,
            "sensitivities":           request.sensitivities or [],
            "allowed_session_minutes": request.allowed_session_minutes or 15,
        }

        profile = pipeline.onboard(ml_onboarding)

        return {
            "success": True,
            "child_profile": profile,
        }

    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/api/session/process")
def process_session(request: SessionProcessRequest):
    """
    Run the full ML inference pipeline for a completed game session.

    Pipeline steps (inside AdaptiveLearningPipeline.process_session):
      1. SkillMasteryClassifier  — per-skill mastery probabilities (XGBoost)
      2. EngagementClassifier    — fatigued / neutral / engaged  (XGBoost)
      3. update_skill_graph      — formula-based skill graph update
      4. decide_next_action      — formula-based next activity selection
      5. generate_parent_summary — human-readable parent summary
      6. generate_session_summary— session stats

    Returns
    -------
    {
      "success": true,
      "pipeline_meta": {...},
      "skill_graph": {...},
      "engagement_state": {...},
      "next_activity": {...},
      "parent_summary": {...},
      "session_summary": {...}
    }
    """
    try:
        session_meta = None
        if request.session_meta:
            session_meta = {
                "session_id":      request.session_meta.session_id or str(uuid.uuid4()),
                "elapsed_seconds": request.session_meta.elapsed_seconds,
                "allowed_minutes": request.session_meta.allowed_minutes,
            }

        output = pipeline.process_session(
            session_events=request.session_events,
            child_profile=request.child_profile,
            session_meta=session_meta,
        )

        return {"success": True, **output}

    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
