import { createContext, useContext, useState, useCallback, useRef } from "react";
import { useNavigate } from "react-router-dom";

const TransitionContext = createContext(null);

/**
 * Wraps <Routes> — when navigateWithFade() is called:
 *   1. Fades the wrapper to opacity 0 (180ms)
 *   2. Calls navigate() — new route mounts while wrapper is still at 0
 *   3. Double-rAF ensures the new DOM is painted before opacity snaps back to 1
 *   4. CSS transition fades the new page in (180ms)
 */
export function TransitionProvider({ children }) {
  const [opacity, setOpacity] = useState(1);
  const navigate = useNavigate();
  const timer = useRef(null);

  const navigateWithFade = useCallback((to, options) => {
    if (timer.current) clearTimeout(timer.current);
    setOpacity(0);
    timer.current = setTimeout(() => {
      navigate(to, options);
      requestAnimationFrame(() => {
        requestAnimationFrame(() => setOpacity(1));
      });
    }, 180);
  }, [navigate]);

  return (
    <TransitionContext.Provider value={{ navigateWithFade }}>
      <div style={{ opacity, transition: "opacity 0.18s ease" }}>
        {children}
      </div>
    </TransitionContext.Provider>
  );
}

/** Drop-in replacement for useNavigate() that includes the fade transition. */
export function usePageNavigate() {
  return useContext(TransitionContext).navigateWithFade;
}
