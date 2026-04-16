import { useState, useEffect } from 'react'
import { Routes, Route } from 'react-router-dom'
import TaxonomyMenu from './TaxonomyMenu'
import RandomForestViz from './RandomForestViz'
import AdaBoostViz from './AdaBoostViz'
import About from './About'
import { C } from './theme'
import { TransitionProvider } from './PageTransition'

function MobileOverlay() {
  const [dismissed, setDismissed] = useState(false)

  if (dismissed) return null

  return (
    <div style={{
      position: 'fixed', inset: 0, zIndex: 9999,
      background: C.bg,
      display: 'flex', flexDirection: 'column',
      alignItems: 'center', justifyContent: 'center',
      padding: '32px 28px',
    }}>
      {/* Logo */}
      <span style={{
        fontWeight: 900, fontSize: 32, letterSpacing: '-0.5px',
        background: `linear-gradient(135deg, ${C.accent}, ${C.green})`,
        WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent',
        backgroundClip: 'text', marginBottom: 32,
      }}>
        Trees
      </span>

      {/* Icon */}
      <svg width="56" height="52" viewBox="0 0 56 52" fill="none" style={{ marginBottom: 28, opacity: 0.35 }}>
        <circle cx="28" cy="11" r="9" stroke={C.dim} strokeWidth="1.8" />
        <circle cx="11" cy="39" r="8" stroke={C.dim} strokeWidth="1.8" />
        <circle cx="45" cy="39" r="8" stroke={C.dim} strokeWidth="1.8" />
        <line x1="28" y1="20" x2="14" y2="31" stroke={C.dim} strokeWidth="1.5" />
        <line x1="28" y1="20" x2="42" y2="31" stroke={C.dim} strokeWidth="1.5" />
      </svg>

      {/* Message */}
      <p style={{
        fontSize: 15, color: C.text, fontWeight: 600,
        textAlign: 'center', lineHeight: 1.7, margin: '0 0 10px',
        maxWidth: 320,
      }}>
        Trees is designed for desktop.
      </p>
      <p style={{
        fontSize: 12, color: C.dim,
        textAlign: 'center', lineHeight: 1.8, margin: '0 0 40px',
        maxWidth: 300,
      }}>
        For the best experience, please visit on a laptop or larger screen.
      </p>

      {/* Continue anyway */}
      <button
        onClick={() => setDismissed(true)}
        style={{
          background: 'none', border: 'none', cursor: 'pointer',
          fontSize: 11, color: C.dimmer,
          fontFamily: 'inherit', padding: '6px 0',
          textDecoration: 'underline', textDecorationColor: 'rgba(255,255,255,0.12)',
          textUnderlineOffset: 3,
        }}
      >
        Continue anyway
      </button>
    </div>
  )
}

export default function App() {
  const [isMobile, setIsMobile] = useState(() => window.innerWidth < 900)

  useEffect(() => {
    const check = () => setIsMobile(window.innerWidth < 900)
    window.addEventListener('resize', check)
    return () => window.removeEventListener('resize', check)
  }, [])

  return (
    <>
      {isMobile && <MobileOverlay />}
      {!isMobile && (
        <TransitionProvider>
          <Routes>
            <Route path="/" element={<TaxonomyMenu />} />
            <Route path="/decision-tree" element={<RandomForestViz mode="decision-tree" />} />
            <Route path="/bagging"       element={<RandomForestViz mode="bagging" />} />
            <Route path="/random-forest" element={<RandomForestViz mode="random-forest" />} />
            <Route path="/adaboost"      element={<AdaBoostViz />} />
            <Route path="/about"         element={<About />} />
          </Routes>
        </TransitionProvider>
      )}
    </>
  )
}
