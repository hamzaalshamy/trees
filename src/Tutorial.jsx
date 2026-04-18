import { createContext, useCallback, useContext, useEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";
import { usePageNavigate } from "./PageTransition";
import { C } from "./theme";
import { TOURS } from "./tutorialSteps";

const MONO = "'JetBrains Mono',monospace";
const CARD_W = 380;
const SPOTLIGHT_PAD = 8;   // px padding around the spotlight rect
const CARD_GAP = 16;       // px gap between spotlight edge and card edge

// ─── Context ──────────────────────────────────────────────────────────────────

const TutorialContext = createContext({
  rfRef: { current: null },
  active: false,
  stepIdx: 0,
  startTutorial: () => {},
  next: () => {},
  back: () => {},
  skip: () => {},
});

// eslint-disable-next-line react-refresh/only-export-components
export const useTutorial = () => useContext(TutorialContext);

// ─── Spotlight + coachmark overlay ────────────────────────────────────────────

function TutorialOverlay({ steps, stepIdx, onNext, onBack, onSkip }) {
  const step = steps[stepIdx];
  const isFirst = stepIdx === 0;
  const isLast  = stepIdx === steps.length - 1;

  // On keyboard-driven steps the Next button is hidden — advance is via → key
  const isKeyboardStep = !!step.keyboardStep;

  // Track target element rect with RAF for smooth repositioning
  const [spotRect, setSpotRect] = useState(null);
  const rafRef = useRef(null);

  useEffect(() => {
    const update = () => {
      if (step.target) {
        const el = document.querySelector(`[data-tutorial="${step.target}"]`);
        if (el) {
          const r = el.getBoundingClientRect();
          setSpotRect({ left: r.left, top: r.top, width: r.width, height: r.height });
        } else {
          setSpotRect(null);
        }
      } else {
        setSpotRect(null);
      }
      rafRef.current = requestAnimationFrame(update);
    };
    rafRef.current = requestAnimationFrame(update);
    return () => {
      if (rafRef.current) cancelAnimationFrame(rafRef.current);
    };
  }, [step.target]);

  // Compute card position
  const cardStyle = (() => {
    if (!spotRect || step.position === "center") {
      // Centered in viewport
      return {
        position: "fixed",
        top: "50%",
        left: "50%",
        transform: "translate(-50%, -50%)",
        width: CARD_W,
      };
    }

    const vw = window.innerWidth;
    const vh = window.innerHeight;
    const sL = spotRect.left - SPOTLIGHT_PAD;
    const sT = spotRect.top  - SPOTLIGHT_PAD;
    const sR = spotRect.left + spotRect.width  + SPOTLIGHT_PAD;
    const sB = spotRect.top  + spotRect.height + SPOTLIGHT_PAD;
    const sCx = (sL + sR) / 2;
    const sCy = (sT + sB) / 2;

    let left, top;
    const pref = step.position;

    const fitsBelow = sB + CARD_GAP + 180 < vh;
    const fitsAbove = sT - CARD_GAP - 180 > 0;
    const fitsRight = sR + CARD_GAP + CARD_W < vw;
    const fitsLeft  = sL - CARD_GAP - CARD_W > 0;

    const CARD_EST_H = 200;

    if (pref === "bottom" && fitsBelow) {
      top  = sB + CARD_GAP;
      left = Math.max(12, Math.min(vw - CARD_W - 12, sCx - CARD_W / 2));
    } else if (pref === "top" && fitsAbove) {
      top  = sT - CARD_GAP - CARD_EST_H;
      left = Math.max(12, Math.min(vw - CARD_W - 12, sCx - CARD_W / 2));
    } else if (pref === "right" && fitsRight) {
      left = sR + CARD_GAP;
      top  = Math.max(12, Math.min(vh - CARD_EST_H - 12, sCy - CARD_EST_H / 2));
    } else if (pref === "left" && fitsLeft) {
      left = sL - CARD_GAP - CARD_W;
      top  = Math.max(12, Math.min(vh - CARD_EST_H - 12, sCy - CARD_EST_H / 2));
    } else {
      // Flip to whichever side has the most room
      if (fitsBelow)      { top = sB + CARD_GAP; left = Math.max(12, Math.min(vw - CARD_W - 12, sCx - CARD_W / 2)); }
      else if (fitsAbove) { top = sT - CARD_GAP - CARD_EST_H; left = Math.max(12, Math.min(vw - CARD_W - 12, sCx - CARD_W / 2)); }
      else if (fitsRight) { left = sR + CARD_GAP; top = Math.max(12, Math.min(vh - CARD_EST_H - 12, sCy - CARD_EST_H / 2)); }
      else                { left = sL - CARD_GAP - CARD_W; top = Math.max(12, Math.min(vh - CARD_EST_H - 12, sCy - CARD_EST_H / 2)); }
    }

    return { position: "fixed", left, top, width: CARD_W };
  })();

  return createPortal(
    // Outer wrapper: pointer-events none so the spotlighted element beneath stays clickable.
    // Only the coachmark card itself captures pointer events.
    <div style={{ position: "fixed", inset: 0, zIndex: 10000, pointerEvents: "none" }}>
      {/* Spotlight rectangle — box-shadow dims everything outside the cutout.
          pointer-events: none so clicks pass through to the element underneath. */}
      {spotRect && (
        <div style={{
          position: "fixed",
          left:   spotRect.left - SPOTLIGHT_PAD,
          top:    spotRect.top  - SPOTLIGHT_PAD,
          width:  spotRect.width  + SPOTLIGHT_PAD * 2,
          height: spotRect.height + SPOTLIGHT_PAD * 2,
          borderRadius: 10,
          boxShadow: [
            "0 0 0 9999px rgba(0,0,0,0.75)",
            `0 0 0 2px ${C.accent}`,
            "0 0 24px rgba(255,165,0,0.35)",
          ].join(", "),
          pointerEvents: "none",
          zIndex: 10001,
        }} />
      )}

      {/* Dim background for centered cards (no spotlight).
          pointer-events: none so underlying page remains interactive. */}
      {!spotRect && (
        <div style={{
          position: "fixed", inset: 0,
          background: "rgba(0,0,0,0.75)",
          pointerEvents: "none",
          zIndex: 10001,
        }} />
      )}

      {/* Coachmark card — the only element that captures pointer events */}
      <div
        style={{
          ...cardStyle,
          zIndex: 10002,
          background: "#12192b",
          border: "1px solid rgba(255,255,255,0.08)",
          borderRadius: 12,
          padding: "20px 22px 16px",
          boxShadow: "0 8px 40px rgba(0,0,0,0.7), 0 0 0 1px rgba(255,255,255,0.04)",
          fontFamily: "Inter, system-ui, sans-serif",
          pointerEvents: "auto",
          userSelect: "none",
        }}
        onClick={e => e.stopPropagation()}
      >
        {/* Step counter */}
        <div style={{ fontSize: 10, color: C.dimmer, fontFamily: MONO, marginBottom: 10, letterSpacing: "0.08em" }}>
          {stepIdx + 1} / {steps.length}
        </div>

        {/* Title */}
        <div style={{ fontSize: 14, fontWeight: 700, color: C.text, marginBottom: 8, lineHeight: 1.3 }}>
          {step.title}
        </div>

        {/* Body */}
        <div style={{ fontSize: 12.5, color: C.dim, lineHeight: 1.65, marginBottom: 18 }}>
          {step.body}
        </div>

        {/* Buttons */}
        <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
          {!isFirst && (
            <button
              onClick={onBack}
              style={{
                padding: "6px 14px", borderRadius: 7,
                border: "1px solid rgba(255,255,255,0.1)",
                background: "transparent", color: C.dim,
                fontSize: 11, fontFamily: "inherit", cursor: "pointer",
                transition: "color 0.15s, border-color 0.15s",
              }}
              onMouseEnter={e => { e.currentTarget.style.color = C.text; e.currentTarget.style.borderColor = "rgba(255,255,255,0.25)"; }}
              onMouseLeave={e => { e.currentTarget.style.color = C.dim; e.currentTarget.style.borderColor = "rgba(255,255,255,0.1)"; }}
            >
              ← Back
            </button>
          )}

          <div style={{ flex: 1 }} />

          <button
            onClick={onSkip}
            style={{
              padding: "6px 12px", borderRadius: 7,
              border: "none", background: "transparent",
              color: C.dimmer, fontSize: 11,
              fontFamily: "inherit", cursor: "pointer",
              transition: "color 0.15s",
            }}
            onMouseEnter={e => { e.currentTarget.style.color = C.dim; }}
            onMouseLeave={e => { e.currentTarget.style.color = C.dimmer; }}
          >
            Skip
          </button>

          {/* On keyboard-driven steps show a dimmed hint instead of Next */}
          {isKeyboardStep ? (
            <span style={{
              padding: "7px 14px", borderRadius: 7,
              background: "rgba(255,255,255,0.04)",
              color: C.dimmer, fontSize: 11, fontFamily: MONO,
              letterSpacing: "0.04em",
            }}>
              press →
            </span>
          ) : (
            <button
              onClick={onNext}
              style={{
                padding: "7px 18px", borderRadius: 7, border: "none",
                background: `linear-gradient(135deg, ${C.accent}, #d97706)`,
                color: "#000", fontSize: 11, fontFamily: "inherit",
                cursor: "pointer", fontWeight: 700,
                transition: "filter 0.15s, box-shadow 0.15s",
              }}
              onMouseEnter={e => { e.currentTarget.style.filter = "brightness(1.1)"; e.currentTarget.style.boxShadow = "0 0 12px rgba(245,158,11,0.3)"; }}
              onMouseLeave={e => { e.currentTarget.style.filter = "none"; e.currentTarget.style.boxShadow = "none"; }}
            >
              {isLast ? "Finish" : "Next →"}
            </button>
          )}
        </div>
      </div>
    </div>,
    document.body
  );
}

// ─── Provider ─────────────────────────────────────────────────────────────────

export function TutorialProvider({ children }) {
  const rfRef  = useRef(null);
  const [active,   setActive]   = useState(false);
  const [stepIdx,  setStepIdx]  = useState(0);
  const navigate = usePageNavigate();

  const steps = TOURS.randomForest;

  // Fire the current step's action whenever stepIdx changes
  useEffect(() => {
    if (!active) return;
    const action = steps[stepIdx]?.action;
    if (action) action(rfRef);
  }, [active, stepIdx, steps]);

  const startTutorial = useCallback(() => {
    navigate("/random-forest");
    setStepIdx(0);
    setActive(true);
  }, [navigate]);

  const next = useCallback(() => {
    setStepIdx(i => {
      const nextIdx = i + 1;
      if (nextIdx >= steps.length) {
        setActive(false);
        return 0;
      }
      return nextIdx;
    });
  }, [steps.length]);

  const back = useCallback(() => {
    setStepIdx(i => Math.max(0, i - 1));
  }, []);

  const skip = useCallback(() => {
    setActive(false);
    setStepIdx(0);
  }, []);

  // ── Keyboard-step handler ───────────────────────────────────────────────────
  // When a step has keyboardStep: true, intercept → before RF's own handler,
  // trigger one split animation step, then auto-advance after 600ms.
  const waitingRef = useRef(false); // prevents double-fire if user holds the key

  useEffect(() => {
    if (!active) return;
    const step = steps[stepIdx];
    if (!step?.keyboardStep) return;

    const handleKey = (e) => {
      if (e.key !== "ArrowRight" && e.key !== "ArrowDown") return;
      if (waitingRef.current) return; // already waiting to advance
      e.preventDefault();
      e.stopPropagation();
      waitingRef.current = true;
      rfRef.current?.stepOnce();
      setTimeout(() => {
        waitingRef.current = false;
        next();
      }, 600);
    };

    // capture: true so we intercept before RF's own keydown handler
    window.addEventListener("keydown", handleKey, { capture: true });
    return () => {
      window.removeEventListener("keydown", handleKey, { capture: true });
      waitingRef.current = false;
    };
  }, [active, stepIdx, steps, next]);

  return (
    <TutorialContext.Provider value={{ rfRef, active, stepIdx, startTutorial, next, back, skip }}>
      {children}
      {active && (
        <TutorialOverlay
          steps={steps}
          stepIdx={stepIdx}
          onNext={next}
          onBack={back}
          onSkip={skip}
        />
      )}
    </TutorialContext.Provider>
  );
}
