import { useCallback, useEffect, useRef, useState } from 'react';
import { Link } from 'react-router-dom';
import { AnimatePresence, motion } from 'motion/react';
import confetti from 'canvas-confetti';
import { useHandTracking } from '../hooks/useHandTracking';
import { GestureCursor } from '../components/home/GestureCursor';
import { Mascot } from '../components/home/Mascot';
import { MathCube } from '../components/game/MathCube';
import { EquationSlot } from '../components/game/EquationSlot';
import { mathEquations } from '../data/mathEquations';
import type { MascotState } from '../data/mascotConfig';

interface CubeNode {
  id: string; value: string;
  colorClass: string; shadowClass: string; textColor: string;
  homeX: number; homeY: number; 
  targetX: number; targetY: number; 
  x: number; y: number; 
  isLocked: boolean; 
}

const COLORS = [
  { bg: 'bg-rose-400', shadow: '#be123c', text: 'text-white' },
  { bg: 'bg-sky-400', shadow: '#0369a1', text: 'text-white' },
  { bg: 'bg-yellow-400', shadow: '#a16207', text: 'text-yellow-950' },
  { bg: 'bg-green-400', shadow: '#15803d', text: 'text-white' },
  { bg: 'bg-purple-400', shadow: '#6b21a8', text: 'text-white' },
];

export function MathEquationRoute() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const { cursor, isReady: handTrackingReady } = useHandTracking(videoRef);

  const [score, setScore] = useState(0);
  const [timeLeft, setTimeLeft] = useState(180); // 3 Minutes timer
  const [difficulty, setDifficulty] = useState<'easy' | 'medium' | 'hard'>('easy');
  const [equationIndex, setEquationIndex] = useState(0); 
  const [gameStatus, setGameStatus] = useState<'playing' | 'win' | 'gameover'>('playing');
  const [cameraError, setCameraError] = useState(false); // NEW: Camera error detection
  
  const [cubes, setCubes] = useState<CubeNode[]>([]);
  const [slotsFilled, setSlotsFilled] = useState<boolean[]>([]);
  const [wrongShakeSlot, setWrongShakeSlot] = useState<number | null>(null);
  const [isCorrectGlow, setIsCorrectGlow] = useState(false);
  
  const [mascotState, setMascotState] = useState<MascotState>('idle');
  const [mascotLine, setMascotLine] = useState('Build the equation!');
  const [mascotNonce, setMascotNonce] = useState(0);

  const [hoveredCubeId, setHoveredCubeId] = useState<string | null>(null);
  const [draggedCubeId, setDraggedCubeId] = useState<string | null>(null);

  const availableEqs = mathEquations.filter(e => e.level === difficulty);
  const currentWord = availableEqs[equationIndex % availableEqs.length];

  // Engine Refs
  const nodesRef = useRef<Record<string, CubeNode>>({});
  const domRefs = useRef<Record<string, HTMLDivElement | null>>({});
  const animationFrameRef = useRef<number | null>(null);
  
  const slotPositions = useRef<{x: number, y: number, expected: string, isFilled: boolean}[]>([]);
  
  // Hand Tracking Smoothing Refs
  const draggedIdRef = useRef<string | null>(null);
  const hoveredIdRef = useRef<string | null>(null);
  const pinchLossFramesRef = useRef(0);
  const cursorRef = useRef(cursor);
  const audioContextRef = useRef<AudioContext | null>(null);
  const isWinningRef = useRef(false);
  const timerRef = useRef<number | null>(null);

  const playSound = useCallback((type: 'ding' | 'wrong' | 'win' | 'pick' | 'gameover') => {
    if (typeof window === 'undefined') return;
    const AudioCtx = window.AudioContext || (window as any).webkitAudioContext;
    if (!audioContextRef.current && AudioCtx) audioContextRef.current = new AudioCtx();
    const ctx = audioContextRef.current;
    if (!ctx || ctx.state === 'suspended') ctx?.resume();

    const osc = ctx.createOscillator(); const gain = ctx.createGain();
    osc.connect(gain); gain.connect(ctx.destination);

    if (type === 'pick') {
      osc.type = 'sine'; osc.frequency.setValueAtTime(400, ctx.currentTime);
      osc.frequency.exponentialRampToValueAtTime(600, ctx.currentTime + 0.1);
      gain.gain.setValueAtTime(0.1, ctx.currentTime); gain.gain.exponentialRampToValueAtTime(0.01, ctx.currentTime + 0.1);
      osc.start(ctx.currentTime); osc.stop(ctx.currentTime + 0.1);
    } else if (type === 'ding') {
      osc.type = 'sine'; osc.frequency.setValueAtTime(800, ctx.currentTime);
      osc.frequency.exponentialRampToValueAtTime(1200, ctx.currentTime + 0.15);
      gain.gain.setValueAtTime(0.3, ctx.currentTime); gain.gain.exponentialRampToValueAtTime(0.01, ctx.currentTime + 0.15);
      osc.start(ctx.currentTime); osc.stop(ctx.currentTime + 0.15);
    } else if (type === 'wrong') {
      osc.type = 'sawtooth'; osc.frequency.setValueAtTime(200, ctx.currentTime);
      osc.frequency.exponentialRampToValueAtTime(100, ctx.currentTime + 0.2);
      gain.gain.setValueAtTime(0.2, ctx.currentTime); gain.gain.linearRampToValueAtTime(0.01, ctx.currentTime + 0.2);
      osc.start(ctx.currentTime); osc.stop(ctx.currentTime + 0.2);
    } else if (type === 'gameover') {
      osc.type = 'square'; osc.frequency.setValueAtTime(150, ctx.currentTime);
      osc.frequency.linearRampToValueAtTime(50, ctx.currentTime + 1);
      gain.gain.setValueAtTime(0.3, ctx.currentTime); gain.gain.linearRampToValueAtTime(0.01, ctx.currentTime + 1);
      osc.start(ctx.currentTime); osc.stop(ctx.currentTime + 1);
    } else if (type === 'win') {
      [523.25, 659.25, 783.99].forEach((freq, i) => {
        const o = ctx.createOscillator(); const g = ctx.createGain();
        o.connect(g); g.connect(ctx.destination);
        o.type = 'triangle'; o.frequency.setValueAtTime(freq, ctx.currentTime + (i * 0.1));
        g.gain.setValueAtTime(0.2, ctx.currentTime + (i * 0.1)); g.gain.exponentialRampToValueAtTime(0.01, ctx.currentTime + (i * 0.1) + 0.6);
        o.start(ctx.currentTime + (i * 0.1)); o.stop(ctx.currentTime + (i * 0.1) + 0.6);
      });
    }
  }, []);

  const speak = useCallback((text: string, state: MascotState) => {
    setMascotLine(text); setMascotState(state); setMascotNonce(n => n + 1);
    if (window.speechSynthesis) {
      window.speechSynthesis.cancel();
      const utterance = new SpeechSynthesisUtterance(text);
      utterance.pitch = 1.3; utterance.rate = 1.1;
      utterance.onend = () => { setMascotState('idle'); setMascotNonce(n => n + 1); };
      window.speechSynthesis.speak(utterance);
    }
  }, []);

  const startRound = useCallback(() => {
    setGameStatus('playing');
    setIsCorrectGlow(false);
    setTimeLeft(180); 
    
    if (timerRef.current) clearInterval(timerRef.current);
    timerRef.current = window.setInterval(() => {
      setTimeLeft(t => {
        if (t <= 1) {
          clearInterval(timerRef.current!);
          setGameStatus('gameover');
          playSound('gameover');
          speak('Oh no! Time is up!', 'sad');
          return 0;
        }
        return t - 1;
      });
    }, 1000);

    const elements = currentWord.elements;
    const shuffled = [...elements].sort(() => 0.5 - Math.random());
    const slotCount = elements.length;
    
    const spacingVw = 14; 
    const slotStartX = 50 - ((slotCount - 1) * spacingVw) / 2;
    const slotY = 40; 

    // Define Slots mathematically
    slotPositions.current = elements.map((expectedValue, i) => ({ 
      x: slotStartX + i * spacingVw, 
      y: slotY,
      expected: expectedValue,
      isFilled: false
    }));

    const trayStartX = 50 - ((slotCount - 1) * spacingVw) / 2;
    const trayY = 82;

    // Build Cubes
    const newNodes: Record<string, CubeNode> = {};
    const newCubes = shuffled.map((value, i) => {
      const color = COLORS[i % COLORS.length];
      const id = `cube-${i}-${value}-${Date.now()}`;
      const homeX = trayStartX + i * spacingVw + (Math.random() * 6 - 3); 
      const homeY = trayY + (Math.random() * 4 - 2);
      
      newNodes[id] = { 
        id, value, colorClass: color.bg, shadowClass: color.shadow, textColor: color.text, 
        homeX, homeY, targetX: homeX, targetY: homeY, x: homeX, y: homeY, 
        isLocked: false 
      };
      return newNodes[id];
    });

    nodesRef.current = newNodes;
    setCubes(newCubes);
    setSlotsFilled(new Array<boolean>(slotCount).fill(false));
    speak('Build the equation!', 'idle');

  }, [currentWord, playSound, speak]);

  useEffect(() => { startRound(); return () => clearInterval(timerRef.current!); }, [startRound]);
  useEffect(() => { cursorRef.current = cursor; }, [cursor]);

  // Main 60FPS ALWAYS-ON Physics Engine
  useEffect(() => {
    if (!handTrackingReady) return;

    const loop = () => {
      const nodes = nodesRef.current;
      const nodeList = Object.values(nodes) as CubeNode[];
      const liveCursor = cursorRef.current;
      
      const vw = window.innerWidth / 100;
      const vh = window.innerHeight / 100;

      // 1. HAND INTERACTION LOGIC
      if (gameStatus === 'playing' && liveCursor) {
        const cx = liveCursor.x * 100; 
        const cy = liveCursor.y * 100;
        
        let isPinching = Boolean(liveCursor.isPinching);
        if (!isPinching) {
          pinchLossFramesRef.current++;
          if (pinchLossFramesRef.current < 20) isPinching = true; 
        } else {
          pinchLossFramesRef.current = 0;
        }

        if (!draggedIdRef.current) {
          let closestId: string | null = null; 
          let minDist = 25 * vh; // Massive forgiving hit radius

          nodeList.forEach(n => {
            if (n.isLocked) return; 
            const nodePxX = n.x * vw;
            const nodePxY = n.y * vh;
            const handPxX = cx * vw;
            const handPxY = cy * vh;
            
            const dist = Math.hypot(nodePxX - handPxX, nodePxY - handPxY);
            if (dist < minDist) { minDist = dist; closestId = n.id; }
          });
          
          hoveredIdRef.current = closestId; 
          setHoveredCubeId(closestId);
        }

        // Execute Grab
        if (isPinching && hoveredIdRef.current && !draggedIdRef.current) {
          draggedIdRef.current = hoveredIdRef.current; 
          setDraggedCubeId(hoveredIdRef.current);
          playSound('pick');
        } 
        
        // Execute Release
        else if (!isPinching && draggedIdRef.current) {
          const node = nodes[draggedIdRef.current];
          let targetSlotIndex: number | null = null;
          let minDist = 20 * vh; // Large forgiving drop radius

          const nodePxX = node.x * vw;
          const nodePxY = node.y * vh;

          slotPositions.current.forEach((slot, index) => {
            if (!slot.isFilled) {
              const slotPxX = slot.x * vw;
              const slotPxY = slot.y * vh;
              const dist = Math.hypot(nodePxX - slotPxX, nodePxY - slotPxY);
              if (dist < minDist) { minDist = dist; targetSlotIndex = index; }
            }
          });

          // PIECE VALIDATION
          if (targetSlotIndex !== null) {
            const slot = slotPositions.current[targetSlotIndex];
            if (node.value === slot.expected) {
              node.targetX = slot.x;
              node.targetY = slot.y;
              node.isLocked = true;
              slot.isFilled = true;
              
              playSound('ding');
              setScore(s => s + 5);
              setSlotsFilled(prev => { const n = [...prev]; n[targetSlotIndex!] = true; return n; });
              
              confetti({ particleCount: 30, spread: 50, origin: { x: slot.x / 100, y: slot.y / 100 }, colors: ['#4ade80', '#fef08a'] });
              speak('Good job!', 'liking-leg');
            } else {
              node.targetX = node.homeX; 
              node.targetY = node.homeY;
              playSound('wrong');
              setWrongShakeSlot(targetSlotIndex);
              setTimeout(() => setWrongShakeSlot(null), 400);
            }
          } else {
            node.targetX = node.homeX; 
            node.targetY = node.homeY;
          }

          draggedIdRef.current = null; setDraggedCubeId(null);
        }
      } 
      // Hand out of frame failsafe
      else if (!liveCursor && draggedIdRef.current) {
        const node = nodes[draggedIdRef.current];
        if (node) {
          node.targetX = node.homeX;
          node.targetY = node.homeY;
        }
        draggedIdRef.current = null; setDraggedCubeId(null);
        hoveredIdRef.current = null; setHoveredCubeId(null);
      }

      // 2. ALWAYS-ON PHYSICS LOGIC
      let lockedCount = 0;

      nodeList.forEach(node => {
        // Instant Snap Alignment
        if (node.isLocked) {
          node.x = node.targetX;
          node.y = node.targetY;
          lockedCount++;
        } 
        else if (node.id === draggedIdRef.current && liveCursor && gameStatus === 'playing') {
          const cx = liveCursor.x * 100;
          const cy = liveCursor.y * 100;
          node.x += (cx - node.x) * 0.95; 
          node.y += (cy - node.y) * 0.95;
        } else {
          node.x += (node.targetX - node.x) * 0.25;
          node.y += (node.targetY - node.y) * 0.25;
        }

        const el = domRefs.current[node.id];
        if (el) { el.style.left = `${node.x}vw`; el.style.top = `${node.y}vh`; }
      });

      // 3. FULL EQUATION COMPLETION (WIN)
      if (lockedCount === currentWord.elements.length && !draggedIdRef.current && gameStatus === 'playing') {
        setGameStatus('win');
        setIsCorrectGlow(true);
        clearInterval(timerRef.current!);
        playSound('win');
        speak('Amazing! You did it!', 'happy');
        setScore(s => s + 20); 
        
        if (timeLeft > 90) {
          setDifficulty(prev => prev === 'easy' ? 'medium' : 'hard');
        }
        
        const end = Date.now() + 2000;
        const frame = () => {
          confetti({ particleCount: 15, angle: 60, spread: 55, origin: { x: 0 }, colors: ['#fde047', '#4ade80', '#38bdf8'] });
          confetti({ particleCount: 15, angle: 120, spread: 55, origin: { x: 1 }, colors: ['#fde047', '#4ade80', '#38bdf8'] });
          if (Date.now() < end) requestAnimationFrame(frame);
        };
        frame();

        setTimeout(() => {
          setEquationIndex(i => i + 1);
          startRound();
        }, 4000);
      }

      animationFrameRef.current = requestAnimationFrame(loop);
    };

    animationFrameRef.current = requestAnimationFrame(loop);
    return () => { if (animationFrameRef.current) cancelAnimationFrame(animationFrameRef.current); };
  }, [handTrackingReady, gameStatus, currentWord, timeLeft, playSound, speak, startRound]);

  // CRITICAL FIX: Ensure camera starts securely with dimensions to prevent browser throttling
  useEffect(() => {
    navigator.mediaDevices.getUserMedia({ video: { facingMode: 'user', width: { ideal: 640 }, height: { ideal: 480 } } })
      .then(stream => { 
        if (videoRef.current) { 
          videoRef.current.srcObject = stream; 
          videoRef.current.onloadedmetadata = () => {
            videoRef.current?.play().catch(console.error);
          };
        } 
      })
      .catch(err => {
        console.error("Camera Error:", err);
        setCameraError(true);
      });
  }, []);

  const handleRestart = () => {
    setDifficulty('easy');
    setEquationIndex(0);
    setScore(0);
    startRound();
  };

  const formatTime = (seconds: number) => {
    const m = Math.floor(seconds / 60);
    const s = seconds % 60;
    return `${m}:${s.toString().padStart(2, '0')}`;
  };

  return (
    <main className="relative min-h-screen overflow-hidden touch-none bg-sky-200">
      <img src="/gardenbg.jpeg" alt="Garden Theme" className="absolute inset-0 h-full w-full object-cover scale-105 blur-[2px]" />
      <div className="absolute inset-0 bg-gradient-to-b from-sky-400/30 via-transparent to-lime-400/40 backdrop-blur-[1px]" /> 
      
      {/* CRITICAL FIX: Video element MUST have physical dimensions in Tailwind for MediaPipe to capture frames properly */}
      <video ref={videoRef} playsInline muted className="fixed top-0 left-0 w-32 h-32 opacity-0 pointer-events-none z-[-1]" />

      {/* CAMERA ERROR NOTIFICATION */}
      <AnimatePresence>
        {cameraError && (
          <motion.div initial={{ y: -50, opacity: 0 }} animate={{ y: 0, opacity: 1 }} className="absolute top-24 left-0 right-0 flex justify-center z-50">
             <div className="bg-rose-500 text-white font-black px-6 py-4 rounded-3xl shadow-xl text-lg border-4 border-white" style={{ fontFamily: '"Comic Sans MS", cursive' }}>
                ⚠️ Camera access denied! Please allow the camera to play.
             </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Top HUD */}
      <div className="relative z-10 flex justify-between items-center p-6 sm:px-12 pointer-events-auto">
        <Link to="/" className="rounded-full border-[5px] border-white bg-pink-400 px-6 py-3 text-xl font-black text-white shadow-[0_8px_0_rgba(190,24,93,0.8)] hover:translate-y-1 hover:shadow-none transition-all" style={{ fontFamily: '"Comic Sans MS", cursive' }}>
          Back Home
        </Link>
        
        {/* Adaptive Timer UI */}
        <div className="flex items-center gap-4 rounded-3xl border-[5px] border-white bg-white px-6 py-2 shadow-[0_8px_0_rgba(0,0,0,0.1)]">
          <span className="text-3xl">⏱️</span>
          <span className={`text-4xl font-black ${timeLeft < 30 ? 'text-rose-500 animate-pulse' : 'text-slate-800'}`} style={{ fontFamily: '"Comic Sans MS", cursive' }}>
            {formatTime(timeLeft)}
          </span>
          <div className="ml-4 rounded-full bg-violet-100 px-3 py-1 text-sm font-bold text-violet-600 uppercase tracking-wider border-2 border-violet-200">
            {difficulty}
          </div>
        </div>

        <div className="flex items-center gap-4 rounded-3xl border-[5px] border-white bg-yellow-300 px-6 py-2 shadow-[0_8px_0_rgba(161,98,7,0.8)]">
          <span className="text-4xl">⭐</span>
          <span className="text-4xl font-black text-yellow-950" style={{ fontFamily: '"Comic Sans MS", cursive' }}>{score}</span>
        </div>
      </div>

      {/* Center UI Block */}
      <div className="relative flex w-full flex-col items-center pt-2">
        <div className="rounded-[40px] border-[8px] border-white bg-violet-500/90 px-12 py-6 shadow-[0_15px_0_rgba(76,29,149,0.8)] backdrop-blur-sm">
           <h2 className="text-4xl font-black text-white text-center" style={{ fontFamily: '"Comic Sans MS", cursive', WebkitTextStroke: '2px rgba(0,0,0,0.2)' }}>
             Build the Equation!
           </h2>
        </div>
      </div>

      <style dangerouslySetInnerHTML={{__html: `
        @keyframes shake {
          0%, 100% { transform: translateX(0); }
          25% { transform: translateX(-15px) rotate(-4deg); }
          50% { transform: translateX(15px) rotate(4deg); }
          75% { transform: translateX(-15px) rotate(-4deg); }
        }
      `}} />

      {/* ABSOLUTE SLOT ALIGNMENT FIX */}
      <div className="absolute inset-0 pointer-events-none z-10">
        {slotPositions.current.map((slot, i) => (
          <div 
            key={`slot-${i}`} 
            className="absolute"
            style={{ left: `${slot.x}vw`, top: `${slot.y}vh`, transform: 'translate(-50%, -50%)' }}
          >
            <EquationSlot isFilled={slotsFilled[i]} isWrongShake={wrongShakeSlot === i} isCorrectGlow={isCorrectGlow} />
          </div>
        ))}
      </div>

      {/* Draggable Cubes */}
      <div className="absolute inset-0 z-20 pointer-events-none">
        <AnimatePresence>
          {cubes.map((cube) => (
            <MathCube
              key={cube.id}
              ref={(el) => { domRefs.current[cube.id] = el; }}
              value={cube.value}
              colorClass={cube.colorClass}
              shadowClass={cube.shadowClass}
              textColor={cube.textColor}
              isHovered={hoveredCubeId === cube.id}
              isDragging={draggedCubeId === cube.id}
              isLocked={cube.isLocked}
              initialX={cube.x}
              initialY={cube.y}
            />
          ))}
        </AnimatePresence>
      </div>

      {/* WIN OVERLAY WITH TROPHY */}
      <AnimatePresence>
        {gameStatus === 'win' && (
          <motion.div 
            initial={{ opacity: 0, scale: 0.5, y: 50 }}
            animate={{ opacity: 1, scale: 1, y: 0 }}
            exit={{ opacity: 0, scale: 0.5, y: -50 }}
            transition={{ type: 'spring', bounce: 0.6 }}
            className="absolute inset-0 z-[60] flex items-center justify-center bg-black/40 backdrop-blur-md pointer-events-auto"
          >
            <div className="flex flex-col items-center rounded-[50px] border-[10px] border-white bg-gradient-to-b from-yellow-300 to-orange-400 p-12 shadow-[0_25px_0_rgba(154,52,18,0.8)]">
              <span className="text-[160px] drop-shadow-2xl animate-bounce">🏆</span>
              <h2 className="mt-4 text-7xl font-black text-white" style={{ fontFamily: '"Comic Sans MS", cursive', WebkitTextStroke: '4px #9a3412', textShadow: '0 8px 0 rgba(0,0,0,0.2)' }}>
                YOU WIN!
              </h2>
              <div className="mt-6 rounded-full border-4 border-white bg-sky-400 px-8 py-3 shadow-[0_6px_0_rgba(2,132,199,0.8)]">
                 <p className="text-3xl font-bold text-white" style={{ fontFamily: '"Comic Sans MS", cursive' }}>+20 Stars! ⭐</p>
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* GAME OVER OVERLAY */}
      <AnimatePresence>
        {gameStatus === 'gameover' && (
          <motion.div 
            initial={{ opacity: 0, scale: 0.5, y: -50 }}
            animate={{ opacity: 1, scale: 1, y: 0 }}
            className="absolute inset-0 z-[60] flex items-center justify-center bg-black/60 backdrop-blur-md pointer-events-auto"
          >
            <div className="flex flex-col items-center rounded-[50px] border-[10px] border-white bg-gradient-to-b from-rose-400 to-rose-600 p-12 shadow-[0_25px_0_rgba(159,18,57,0.8)] text-center">
              <span className="text-[120px] drop-shadow-2xl">⏰</span>
              <h2 className="mt-4 text-6xl font-black text-white" style={{ fontFamily: '"Comic Sans MS", cursive', WebkitTextStroke: '3px #881337', textShadow: '0 8px 0 rgba(0,0,0,0.2)' }}>
                TIME'S UP!
              </h2>
              <p className="mt-4 text-2xl font-bold text-rose-100">Let's try an easier one!</p>
              <button 
                onClick={handleRestart}
                className="mt-8 rounded-full border-4 border-white bg-yellow-400 px-10 py-4 text-3xl font-black text-yellow-950 shadow-[0_8px_0_rgba(161,98,7,0.8)] hover:translate-y-1 hover:shadow-none transition-all"
                style={{ fontFamily: '"Comic Sans MS", cursive' }}
              >
                Play Again!
              </button>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      <Mascot state={mascotState} line={mascotLine} nonce={mascotNonce} />
      
      {/* Fallback to ensure cursor shows even if hand tracking state glitches slightly */}
      <GestureCursor cursor={cursor} holdProgress={0} isVisible={handTrackingReady || Boolean(cursor)} isActive={Boolean(draggedCubeId)} accentColor="#facc15" centerColor={draggedCubeId ? '#facc15' : '#fef08a'} size={draggedCubeId ? 'large' : 'default'} />
    </main>
  );
}