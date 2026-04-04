// =============================================================================
// src/services/api.ts
//
// API service layer for the PlaySpark Adaptive Learning backend.
//
// All calls are routed through the Vite dev proxy (/api/* → localhost:8000),
// so no CORS configuration is needed on the frontend side.
//
// FALLBACK GUARANTEE: Every exported function returns null (never throws)
// when the backend is unavailable. Callers MUST handle null and show
// hardcoded defaults. The UI must never break due to a backend outage.
// =============================================================================

// Vite proxies /api/* to http://localhost:8000 (configured in vite.config.ts).
// Leave this empty string — do NOT add a hostname here.
const API_BASE = '';

const REQUEST_TIMEOUT_MS = 5_000;   // onboarding
const SESSION_TIMEOUT_MS  = 8_000;  // session processing (heavier pipeline)

// ---------------------------------------------------------------------------
// Type definitions — mirrors the Python Pydantic / JSON schema models
// ---------------------------------------------------------------------------

export interface OnboardingPayload {
  /** Parent email (collected at onboarding step 1) */
  parent_email: string;
  /** Integer age 3-7 (collected at onboarding step 2) */
  child_age: number;
  /** Interest IDs from step 3: 'music' | 'stories' | 'puzzles' | 'art' */
  interests: string[];
  /** Goal IDs from step 4: 'phonics' | 'counting' | 'colors' | 'nature' */
  goals: string[];
  child_name?: string;
  language?: string;
  sensitivities?: string[];
  allowed_session_minutes?: number;
}

export interface FirstSessionPlan {
  structure: string[];
  warm_up_skill: string;
  main_skill: string;
  cool_down_skill: string;
  starting_difficulty: number;
  preferred_modality: string;
  session_minutes: number;
  tasks_per_skill: number;
  hint_mode: string;
  reward_frequency: string;
  note: string;
}

/** Full ChildProfile as returned by /api/onboard (matches starter_profile_schema.json) */
export interface ChildProfile {
  child_id: string;
  name: string;
  age_months: number;
  age_band: string;
  language: string;
  interests: string[];
  goals: string[];
  sessions_completed: number;
  preferred_modality: string;
  allowed_session_minutes: number;
  current_difficulty_per_skill: Record<string, number>;
  skill_mastery_probs: Record<string, number>;
  unlocked_skills: string[];
  recommended_first_session_plan: FirstSessionPlan;
  flags: {
    new_user: boolean;
    needs_calibration: boolean;
    cold_start: boolean;
  };
  safe_difficulty_range: {
    min: number;
    max: number;
    min_label: string;
    max_label: string;
  };
  created_at: string;
  updated_at: string;
}

/** Full ML pipeline output returned by /api/session/process */
export interface SessionResult {
  success: boolean;
  pipeline_meta: {
    child_id: string;
    session_id: string;
    processed_at: string;
    inference_ms: number;
    models_trained: { skill_mastery: boolean; engagement: boolean };
    n_events: number;
  };
  skill_graph: Record<string, unknown>;
  engagement_state: {
    engagement_state: 'engaged' | 'neutral' | 'fatigued';
    fatigue_risk: string;
    alert_level: string;
    cold_start: boolean;
    [key: string]: unknown;
  };
  next_activity: {
    next_skill_focus: string;
    next_difficulty_label: string;
    next_modality: string;
    hint_intensity: string;
    reward_type: string;
    reason: string;
    parent_tip?: string;
    [key: string]: unknown;
  };
  parent_summary: {
    human_summary: string;
    strengths: Array<{ skill_name: string; display_name: string }>;
    weak_areas: Array<{ skill_name: string; display_name: string }>;
    recommended_offline_activity: string;
    [key: string]: unknown;
  };
  session_summary: {
    duration_minutes: number;
    tasks_completed: number;
    overall_accuracy: number;
    summary_for_parent: string;
    [key: string]: unknown;
  };
}

// ---------------------------------------------------------------------------
// Internal: race a fetch against a timeout
// ---------------------------------------------------------------------------
function withTimeout<T>(promise: Promise<T>, ms: number): Promise<T> {
  return Promise.race([
    promise,
    new Promise<never>((_, reject) =>
      setTimeout(() => reject(new Error(`Request timed out after ${ms}ms`)), ms)
    ),
  ]);
}

// ---------------------------------------------------------------------------
// POST /api/onboard
// ---------------------------------------------------------------------------

/**
 * Submit onboarding form data to the ML backend and receive a ChildProfile.
 *
 * Returns the profile on success, or **null** if:
 *   - The backend is not running
 *   - The request times out (> 5 s)
 *   - The server returns a non-2xx status
 *
 * Callers MUST handle null and fall back to hardcoded defaults.
 */
export async function callOnboardingAPI(
  payload: OnboardingPayload
): Promise<ChildProfile | null> {
  try {
    const response = await withTimeout(
      fetch(`${API_BASE}/api/onboard`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      }),
      REQUEST_TIMEOUT_MS
    );

    if (!response.ok) {
      console.warn(`[PlaySpark API] /api/onboard → HTTP ${response.status}. Falling back to defaults.`);
      return null;
    }

    const data = await response.json() as { success: boolean; child_profile?: ChildProfile };
    return data.child_profile ?? null;

  } catch (err) {
    console.warn('[PlaySpark API] /api/onboard unavailable — using hardcoded fallback profile.', err);
    return null;
  }
}

// ---------------------------------------------------------------------------
// POST /api/session/process
// ---------------------------------------------------------------------------

/**
 * Send completed session events + current child profile to the ML pipeline.
 * Returns the full SessionResult, or **null** on failure.
 *
 * Callers should display hardcoded "good job" messaging when null is returned.
 */
export async function callSessionProcessAPI(
  sessionEvents: Record<string, unknown>[],
  childProfile: ChildProfile,
  sessionMeta?: {
    session_id?: string;
    elapsed_seconds?: number;
    allowed_minutes?: number;
  }
): Promise<SessionResult | null> {
  try {
    const response = await withTimeout(
      fetch(`${API_BASE}/api/session/process`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          session_events: sessionEvents,
          child_profile: childProfile,
          session_meta: sessionMeta ?? null,
        }),
      }),
      SESSION_TIMEOUT_MS
    );

    if (!response.ok) {
      console.warn(`[PlaySpark API] /api/session/process → HTTP ${response.status}. Falling back.`);
      return null;
    }

    return (await response.json()) as SessionResult;

  } catch (err) {
    console.warn('[PlaySpark API] /api/session/process unavailable.', err);
    return null;
  }
}
