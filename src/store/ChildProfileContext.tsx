// =============================================================================
// src/store/ChildProfileContext.tsx
//
// Global React context that stores the ML-generated ChildProfile produced
// during onboarding.  May be null if the backend was unavailable — all
// consumers MUST check for null and use their own hardcoded fallbacks.
// =============================================================================

import { createContext, useContext, useState, type ReactNode } from 'react';
import type { ChildProfile } from '../services/api';

interface ChildProfileState {
  /**
   * The ML-generated profile for this session.
   * null  → backend unavailable or onboarding not yet complete.
   */
  profile: ChildProfile | null;
  /**
   * Store the profile (and mark onboarding as complete).
   * Pass null explicitly if the API failed — isProfileLoaded will still
   * become true so callers know onboarding ran.
   */
  setProfile: (profile: ChildProfile | null) => void;
  /**
   * Becomes true after onboarding finishes, regardless of whether the
   * API call succeeded.  Use this to distinguish "not loaded yet" from
   * "loaded but API returned null".
   */
  isProfileLoaded: boolean;
}

const ChildProfileContext = createContext<ChildProfileState | undefined>(undefined);

export function ChildProfileProvider({ children }: { children: ReactNode }) {
  const [profile, setProfileState] = useState<ChildProfile | null>(null);
  const [isProfileLoaded, setIsProfileLoaded] = useState(false);

  const setProfile = (p: ChildProfile | null) => {
    setProfileState(p);
    setIsProfileLoaded(true);
  };

  return (
    <ChildProfileContext.Provider value={{ profile, setProfile, isProfileLoaded }}>
      {children}
    </ChildProfileContext.Provider>
  );
}

/**
 * Access the child profile from any component inside <ChildProfileProvider>.
 * Always check `profile !== null` before using profile fields.
 */
export function useChildProfile(): ChildProfileState {
  const ctx = useContext(ChildProfileContext);
  if (!ctx) {
    throw new Error('useChildProfile must be used inside <ChildProfileProvider>');
  }
  return ctx;
}
