import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { AnimatePresence, motion } from 'motion/react';
import { useHandTracking } from '../../hooks/useHandTracking';
import { GestureCursor } from './GestureCursor';
import { Mascot } from './Mascot';
import { SubjectCard } from './SubjectCard';
import {
  MascotDialogueGroup,
  MascotState,
  mascotDialogues,
} from '../../data/mascotConfig';
import { subjects } from '../../data/subjects';
import { useChildProfile } from '../../store/ChildProfileContext';

// Maps ML skill names → child-friendly display labels for the mascot greeting
const SKILL_GREET_LABELS: Record<string, string> = {
  phonics:               'letters',
  letter_recognition:    'the alphabet',
  counting:              'counting',
  number_order:          'numbers',
  shape_recognition:     'shapes',
  color_sorting:         'colours',
  hand_eye_coordination: 'coordination',
  attention_retention:   'focus',
  memory_recall:         'memory',
  social_response:       'social skills',
  emotional_regulation:  'feelings',
};

const HOLD_DURATION_MS = 1000;
const SPEECH_RATE = 0.9;
const SPEECH_PITCH = 1.1;

function pickRandom<T>(items: T[]) {
  return items[Math.floor(Math.random() * items.length)];
}

type HoverTarget =
  | { type: 'subject'; id: string }
  | { type: 'game'; id: string }
  | { type: 'popup-back'; id: 'popup-back' }
  | null;

export function HomeScreen() {
  const navigate = useNavigate();
  const videoRef = useRef<HTMLVideoElement>(null);
  const { cursor, isReady: handTrackingReady } = useHandTracking(videoRef);

  // ML child profile — null when backend is unavailable (fallback mode)
  const { profile } = useChildProfile();

  const [cameraStatus, setCameraStatus] = useState<'initializing' | 'ready' | 'error'>('initializing');
  const [hoveredTarget, setHoveredTarget] = useState<HoverTarget>(null);
  const [holdProgress, setHoldProgress] = useState(0);
  const [selectedSubjectId, setSelectedSubjectId] = useState<string | null>(null);
  const [mascotState, setMascotStateValue] = useState<MascotState>('idle');
  const [mascotLine, setMascotLine] = useState('');
  const [mascotNonce, setMascotNonce] = useState(0);

  const audioContextRef = useRef<AudioContext | null>(null);
  const hoverTimeoutRef = useRef<number | null>(null);
  const holdFrameRef = useRef<number | null>(null);
  const holdTargetRef = useRef<string | null>(null);
  const holdStartedAtRef = useRef(0);
  const didWelcomeRef = useRef(false);
  const hoveredRef = useRef<string | null>(null);
  const speechTimeoutRef = useRef<number | null>(null);
  const mascotResetRef = useRef<number | null>(null);
  const subjectRefs = useRef<Record<string, HTMLButtonElement | null>>({});
  const popupGameRefs = useRef<Record<string, HTMLButtonElement | null>>({});
  const popupBackRef = useRef<HTMLButtonElement | null>(null);
  const previousPinchRef = useRef(false);

  const selectedSubject = useMemo(
    () => subjects.find((subject) => subject.id === selectedSubjectId) ?? null,
    [selectedSubjectId],
  );

  const isHomeReady = cameraStatus === 'ready' && handTrackingReady;

  const clearHoldLoop = useCallback(() => {
    if (holdFrameRef.current) {
      cancelAnimationFrame(holdFrameRef.current);
      holdFrameRef.current = null;
    }
    holdTargetRef.current = null;
    holdStartedAtRef.current = 0;
    setHoldProgress(0);
  }, []);

  const clearMascotTimers = useCallback(() => {
    if (speechTimeoutRef.current) {
      window.clearTimeout(speechTimeoutRef.current);
      speechTimeoutRef.current = null;
    }
    if (mascotResetRef.current) {
      window.clearTimeout(mascotResetRef.current);
      mascotResetRef.current = null;
    }
  }, []);

  const setMascotState = useCallback((state: MascotState, duration?: number) => {
    clearMascotTimers();
    setMascotStateValue(state);
    setMascotNonce((value) => value + 1);

    if (duration) {
      mascotResetRef.current = window.setTimeout(() => {
        setMascotStateValue('idle');
        setMascotNonce((value) => value + 1);
      }, duration);
    }
  }, [clearMascotTimers]);

  const mascotSpeak = useCallback((text: string) => {
    clearMascotTimers();
    setMascotLine(text);
    setMascotStateValue('talking');
    setMascotNonce((value) => value + 1);

    const synth = typeof window !== 'undefined' ? window.speechSynthesis : null;

    if (synth) {
      synth.cancel();
      const utterance = new SpeechSynthesisUtterance(text);
      utterance.rate = SPEECH_RATE;
      utterance.pitch = SPEECH_PITCH;
      utterance.onend = () => {
        setMascotStateValue('idle');
        setMascotNonce((value) => value + 1);
      };
      synth.speak(utterance);
    } else {
      speechTimeoutRef.current = window.setTimeout(() => {
        setMascotStateValue('idle');
        setMascotNonce((value) => value + 1);
      }, 1200);
    }
  }, [clearMascotTimers]);

  const speakDialogue = useCallback((group: MascotDialogueGroup, reaction?: MascotState) => {
    if (reaction) {
      setMascotState(reaction, 550);
      window.setTimeout(() => mascotSpeak(pickRandom(mascotDialogues[group])), 280);
      return;
    }
    mascotSpeak(pickRandom(mascotDialogues[group]));
  }, [mascotSpeak, setMascotState]);

  const ensureAudioContext = useCallback(() => {
    if (typeof window === 'undefined') return null;
    const AudioContextCtor = window.AudioContext || (window as Window & { webkitAudioContext?: typeof AudioContext }).webkitAudioContext;
    if (!AudioContextCtor) return null;
    if (!audioContextRef.current) {
      audioContextRef.current = new AudioContextCtor();
    }
    if (audioContextRef.current.state === 'suspended') {
      void audioContextRef.current.resume();
    }
    return audioContextRef.current;
  }, []);

  const playUiSound = useCallback((type: 'hover' | 'select') => {
    const context = ensureAudioContext();
    if (!context) return;

    const oscillator = context.createOscillator();
    const gainNode = context.createGain();
    oscillator.connect(gainNode);
    gainNode.connect(context.destination);

    if (type === 'hover') {
      oscillator.type = 'triangle';
      oscillator.frequency.setValueAtTime(540, context.currentTime);
      oscillator.frequency.exponentialRampToValueAtTime(700, context.currentTime + 0.12);
      gainNode.gain.setValueAtTime(0.0001, context.currentTime);
      gainNode.gain.exponentialRampToValueAtTime(0.06, context.currentTime + 0.02);
      gainNode.gain.exponentialRampToValueAtTime(0.0001, context.currentTime + 0.15);
      oscillator.start(context.currentTime);
      oscillator.stop(context.currentTime + 0.15);
      return;
    }

    oscillator.type = 'square';
    oscillator.frequency.setValueAtTime(420, context.currentTime);
    oscillator.frequency.exponentialRampToValueAtTime(880, context.currentTime + 0.18);
    gainNode.gain.setValueAtTime(0.0001, context.currentTime);
    gainNode.gain.exponentialRampToValueAtTime(0.1, context.currentTime + 0.02);
    gainNode.gain.exponentialRampToValueAtTime(0.0001, context.currentTime + 0.2);
    oscillator.start(context.currentTime);
    oscillator.stop(context.currentTime + 0.2);
  }, [ensureAudioContext]);

  const openSubjectMenu = useCallback((subjectId: string) => {
    if (hoverTimeoutRef.current) {
      window.clearTimeout(hoverTimeoutRef.current);
      hoverTimeoutRef.current = null;
    }
    setSelectedSubjectId(subjectId);
    playUiSound('select');
    speakDialogue('success', 'happy');
    clearHoldLoop();
    setHoveredTarget(null);
    hoveredRef.current = null;
  }, [clearHoldLoop, playUiSound, setHoveredTarget, speakDialogue]);

  const updateHoverTarget = useCallback((nextTarget: HoverTarget) => {
    const nextHoverId = nextTarget ? `${nextTarget.type}:${nextTarget.id}` : null;
    if (hoveredRef.current === nextHoverId) return;

    hoveredRef.current = nextHoverId;
    setHoveredTarget(nextTarget);
    clearHoldLoop();

    if (hoverTimeoutRef.current) {
      window.clearTimeout(hoverTimeoutRef.current);
      hoverTimeoutRef.current = null;
    }

    if (nextTarget) {
      playUiSound('hover');
      setMascotState('hover', 700);
      hoverTimeoutRef.current = window.setTimeout(() => {
        mascotSpeak(pickRandom(mascotDialogues.hover));
      }, 160);
    } else {
      setMascotState('idle');
    }
  }, [clearHoldLoop, mascotSpeak, playUiSound, setMascotState]);

  const beginHoldLoop = useCallback((targetKey: string) => {
    clearHoldLoop();
    holdTargetRef.current = targetKey;
    holdStartedAtRef.current = performance.now();

    const tick = (time: number) => {
      if (holdTargetRef.current !== targetKey) return;

      const progress = Math.min((time - holdStartedAtRef.current) / HOLD_DURATION_MS, 1);
      setHoldProgress(progress);
      holdFrameRef.current = requestAnimationFrame(tick);
    };

    holdFrameRef.current = requestAnimationFrame(tick);
  }, [clearHoldLoop]);

  useEffect(() => {
    let stream: MediaStream | null = null;
    let alive = true;

    async function setupCamera() {
      if (!videoRef.current) return;
      try {
        const nextStream = await navigator.mediaDevices.getUserMedia({
          video: {
            width: { ideal: 640 },
            height: { ideal: 480 },
            frameRate: { ideal: 30 },
            facingMode: 'user',
          },
        });
        if (!alive || !videoRef.current) {
          nextStream.getTracks().forEach((track) => track.stop());
          return;
        }

        stream = nextStream;
        videoRef.current.srcObject = nextStream;
        videoRef.current.onloadedmetadata = async () => {
          await videoRef.current?.play();
          setCameraStatus('ready');
        };
      } catch (error) {
        console.error('Error accessing webcam on home screen:', error);
        if (alive) setCameraStatus('error');
      }
    }

    setupCamera();

    return () => {
      alive = false;
      stream?.getTracks().forEach((track) => track.stop());
      window.speechSynthesis?.cancel();
      clearHoldLoop();
      clearMascotTimers();
      if (hoverTimeoutRef.current) window.clearTimeout(hoverTimeoutRef.current);
    };
  }, [clearHoldLoop, clearMascotTimers]);

  useEffect(() => {
    if (!isHomeReady || didWelcomeRef.current) return;
    didWelcomeRef.current = true;

    // --- Dynamic greeting (ML profile available) ---
    // Uses the recommended first session plan to tell the child what to focus on.
    // Falls back to the hardcoded mascotDialogues.home string when profile is null
    // (backend unavailable) so the experience is identical to the original.
    if (profile?.recommended_first_session_plan?.main_skill) {
      const skillLabel =
        SKILL_GREET_LABELS[profile.recommended_first_session_plan.main_skill] ??
        profile.recommended_first_session_plan.main_skill;
      mascotSpeak(`Hi! Today let's practise ${skillLabel}! Pick a subject!`);
    } else {
      // Hardcoded fallback — original behaviour
      mascotSpeak(mascotDialogues.home.join(' '));
    }
  }, [isHomeReady, mascotSpeak, profile]);

  useEffect(() => {
    if (selectedSubjectId) {
      clearHoldLoop();
    }

    if (!cursor || !isHomeReady) {
      updateHoverTarget(null);
      return;
    }

    const pointX = cursor.x * window.innerWidth;
    const pointY = cursor.y * window.innerHeight;

    let nextTarget: HoverTarget = null;

    if (selectedSubject) {
      if (popupBackRef.current) {
        const rect = popupBackRef.current.getBoundingClientRect();
        if (pointX >= rect.left && pointX <= rect.right && pointY >= rect.top && pointY <= rect.bottom) {
          nextTarget = { type: 'popup-back', id: 'popup-back' };
        }
      }

      if (!nextTarget) {
        for (const game of selectedSubject.games) {
          const element = popupGameRefs.current[game.id];
          if (!element) continue;
          const rect = element.getBoundingClientRect();
          if (pointX >= rect.left && pointX <= rect.right && pointY >= rect.top && pointY <= rect.bottom) {
            nextTarget = { type: 'game', id: game.id };
            break;
          }
        }
      }
    } else {
      for (const subject of subjects) {
        const element = subjectRefs.current[subject.id];
        if (!element) continue;
        const rect = element.getBoundingClientRect();
        if (pointX >= rect.left && pointX <= rect.right && pointY >= rect.top && pointY <= rect.bottom) {
          nextTarget = { type: 'subject', id: subject.id };
          break;
        }
      }
    }

    updateHoverTarget(nextTarget);
  }, [clearHoldLoop, cursor, isHomeReady, selectedSubject, selectedSubjectId, updateHoverTarget]);

  useEffect(() => {
    if (!hoveredTarget) return;
    beginHoldLoop(`${hoveredTarget.type}:${hoveredTarget.id}`);
    return clearHoldLoop;
  }, [beginHoldLoop, clearHoldLoop, hoveredTarget]);

  const resetSelection = useCallback(() => {
    setSelectedSubjectId(null);
    setHoveredTarget(null);
    setMascotState('idle');
    mascotSpeak('Pick a subject!');
  }, [mascotSpeak, setHoveredTarget, setMascotState]);

  const launchGame = useCallback((route?: string) => {
    if (!route) {
      setMascotState('thinking', 900);
      mascotSpeak(pickRandom(mascotDialogues.error));
      return;
    }

    playUiSound('select');
    navigate(route);
  }, [mascotSpeak, navigate, playUiSound, setMascotState]);

  useEffect(() => {
    const nextIsPinching = Boolean(cursor?.isPinching);
    const didClick = nextIsPinching && !previousPinchRef.current;
    previousPinchRef.current = nextIsPinching;

    if (!didClick || !hoveredTarget) return;

    if (hoveredTarget.type === 'subject') {
      openSubjectMenu(hoveredTarget.id);
      return;
    }

    if (hoveredTarget.type === 'popup-back') {
      resetSelection();
      return;
    }

    if (hoveredTarget.type === 'game' && selectedSubject) {
      const targetGame = selectedSubject.games.find((game) => game.id === hoveredTarget.id);
      launchGame(targetGame?.route);
    }
  }, [cursor?.isPinching, hoveredTarget, launchGame, openSubjectMenu, resetSelection, selectedSubject]);

  const homeTitleStyle = {
    fontFamily: '"Comic Sans MS", "Trebuchet MS", "Marker Felt", sans-serif',
    WebkitTextStroke: '5px #7c3aed',
    textShadow: '0 8px 0 rgba(255, 255, 255, 0.24)',
  } as const;

  return (
    <main className="relative min-h-screen overflow-hidden bg-slate-950 text-white">
      <video
        src="/homebg.mp4"
        autoPlay
        muted
        loop
        playsInline
        className="absolute inset-0 h-full w-full object-cover scale-105 blur-[3px]"
      />
      <div className="absolute inset-0 bg-black/40" />
      <div className="absolute inset-0 bg-[radial-gradient(circle_at_top_left,rgba(255,255,255,0.18),transparent_28%),radial-gradient(circle_at_bottom_left,rgba(34,211,238,0.18),transparent_30%),radial-gradient(circle_at_top_right,rgba(250,204,21,0.18),transparent_26%)]" />

      <video ref={videoRef} playsInline muted className="absolute h-0 w-0 opacity-0 pointer-events-none" />

      <div className="relative z-10 min-h-screen px-5 pb-10 pt-8 sm:px-8 lg:px-12">
        <motion.div
          initial={{ opacity: 0, y: -24 }}
          animate={{ opacity: 1, y: 0 }}
          className="mx-auto flex max-w-6xl flex-col gap-8"
        >
          <div className="max-w-3xl">
            <p
              className="inline-flex rounded-full border-[4px] border-white bg-pink-400 px-4 py-2 text-sm font-black uppercase tracking-[0.25em] text-white shadow-[0_10px_0_rgba(0,0,0,0.15)]"
              style={{ fontFamily: '"Comic Sans MS", "Trebuchet MS", "Marker Felt", sans-serif' }}
            >
              Wave to Choose
            </p>
            <h1
              className="mt-5 text-5xl font-black leading-[0.95] sm:text-7xl lg:text-8xl"
              style={homeTitleStyle}
            >
              PlaySpark
              <br />
              Learning Land
            </h1>
            <p
              className="mt-4 max-w-2xl text-lg font-black text-yellow-100 sm:text-2xl"
              style={{
                fontFamily: '"Comic Sans MS", "Trebuchet MS", "Marker Felt", sans-serif',
                WebkitTextStroke: '2px #7c2d12',
              }}
            >
              Float your finger over a subject card, then pinch in the air to pop it open.
            </p>
          </div>

          <div className="grid gap-5 sm:grid-cols-2 xl:max-w-4xl">
            {subjects.map((subject) => (
              <div key={subject.id} className="contents">
                <SubjectCard
                  subject={subject}
                  isHovered={hoveredTarget?.type === 'subject' && hoveredTarget.id === subject.id}
                  holdProgress={hoveredTarget?.type === 'subject' && hoveredTarget.id === subject.id ? holdProgress : 0}
                  onClick={() => openSubjectMenu(subject.id)}
                  onMouseEnter={() => updateHoverTarget({ type: 'subject', id: subject.id })}
                  onMouseLeave={() => updateHoverTarget(null)}
                  setRef={(element) => {
                    subjectRefs.current[subject.id] = element;
                  }}
                />
              </div>
            ))}
          </div>
        </motion.div>
      </div>

      <AnimatePresence>
        {!isHomeReady && cameraStatus !== 'error' && (
          <motion.div
            initial={{ opacity: 0, y: 14 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: 10 }}
            className="absolute left-6 top-6 z-[60] rounded-[30px] border-[4px] border-white bg-cyan-400 px-5 py-3 shadow-[0_14px_0_rgba(0,0,0,0.18)]"
          >
            <p
              className="text-lg font-black text-slate-900"
              style={{ fontFamily: '"Comic Sans MS", "Trebuchet MS", "Marker Felt", sans-serif' }}
            >
              Camera waking up...
            </p>
          </motion.div>
        )}
      </AnimatePresence>

      <AnimatePresence>
        {cameraStatus === 'error' && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="absolute inset-0 z-[80] flex items-center justify-center bg-slate-950/85 px-6"
          >
            <div className="max-w-xl rounded-[36px] border-[5px] border-white bg-rose-400 p-8 text-center shadow-[0_22px_0_rgba(0,0,0,0.18)]">
              <p
                className="text-3xl font-black text-white"
                style={{ fontFamily: '"Comic Sans MS", "Trebuchet MS", "Marker Felt", sans-serif' }}
              >
                Camera needed!
              </p>
              <p className="mt-4 text-lg font-black text-rose-950">
                Let PlaySpark see your hand so the cards can light up and respond.
              </p>
              <button
                type="button"
                onClick={() => window.location.reload()}
                className="mt-6 rounded-[24px] border-[4px] border-white bg-yellow-300 px-6 py-3 text-lg font-black text-rose-700 shadow-[0_10px_0_rgba(0,0,0,0.16)]"
                style={{ fontFamily: '"Comic Sans MS", "Trebuchet MS", "Marker Felt", sans-serif' }}
              >
                Try Again
              </button>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      <AnimatePresence>
        {selectedSubject && (
          <motion.div
            initial={{ opacity: 0, scale: 0.92 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0, scale: 0.96 }}
            className="absolute inset-0 z-[65] flex items-center justify-center bg-slate-950/65 px-5"
          >
            <div className="w-full max-w-2xl rounded-[42px] border-[5px] border-white bg-white/92 p-6 shadow-[0_26px_0_rgba(0,0,0,0.18)] backdrop-blur-md">
              <div className="flex items-start justify-between gap-4">
                <div>
                  <p
                    className="text-base font-black uppercase tracking-[0.3em]"
                    style={{ color: selectedSubject.accent, fontFamily: '"Comic Sans MS", "Trebuchet MS", "Marker Felt", sans-serif' }}
                  >
                    {selectedSubject.name}
                  </p>
                  <h2
                    className="mt-2 text-4xl font-black text-white"
                    style={{
                      fontFamily: '"Comic Sans MS", "Trebuchet MS", "Marker Felt", sans-serif',
                      WebkitTextStroke: `4px ${selectedSubject.textStroke}`,
                    }}
                  >
                    Choose a game
                  </h2>
                </div>
                <button
                  ref={popupBackRef}
                  type="button"
                  onClick={resetSelection}
                  onMouseEnter={() => updateHoverTarget({ type: 'popup-back', id: 'popup-back' })}
                  onMouseLeave={() => updateHoverTarget(null)}
                  className="rounded-full border-[4px] border-white bg-rose-400 px-4 py-2 text-lg font-black text-white shadow-[0_8px_0_rgba(0,0,0,0.16)]"
                >
                  Back
                </button>
              </div>

              <div className="mt-6 grid gap-4">
                {selectedSubject.games.map((game) => (
                  <motion.button
                    key={game.id}
                    ref={(element) => {
                      popupGameRefs.current[game.id] = element;
                    }}
                    type="button"
                    whileHover={{ scale: 1.03, rotate: -1 }}
                    whileTap={{ scale: 0.98 }}
                    onClick={() => launchGame(game.route)}
                    onMouseEnter={() => updateHoverTarget({ type: 'game', id: game.id })}
                    onMouseLeave={() => updateHoverTarget(null)}
                    animate={{
                      scale: hoveredTarget?.type === 'game' && hoveredTarget.id === game.id ? 1.035 : 1,
                      y: hoveredTarget?.type === 'game' && hoveredTarget.id === game.id ? -4 : 0,
                    }}
                    transition={{ type: 'spring', stiffness: 260, damping: 18 }}
                    className="rounded-[30px] border-[5px] border-white bg-sky-200 p-5 text-left shadow-[0_14px_0_rgba(0,0,0,0.16)]"
                    style={{
                      boxShadow: `0 14px 0 rgba(0, 0, 0, 0.16), 0 0 0 5px ${selectedSubject.accent}`,
                    }}
                  >
                    <div className="flex items-center justify-between gap-4">
                      <div>
                        <h3
                          className="text-2xl font-black text-white"
                          style={{
                            fontFamily: '"Comic Sans MS", "Trebuchet MS", "Marker Felt", sans-serif',
                            WebkitTextStroke: `3px ${selectedSubject.textStroke}`,
                          }}
                        >
                          {game.title}
                        </h3>
                        <p className="mt-2 text-base font-black text-slate-700">{game.description}</p>
                      </div>
                      <div className="rounded-full border-[4px] border-white bg-white px-4 py-2 text-sm font-black uppercase tracking-[0.2em] text-slate-800">
                        {game.status === 'ready' ? 'Play' : 'Soon'}
                      </div>
                    </div>
                  </motion.button>
                ))}
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      <Mascot state={mascotState} line={mascotLine} nonce={mascotNonce} />
      <GestureCursor cursor={cursor} holdProgress={holdProgress} isVisible={isHomeReady} />
    </main>
  );
}
