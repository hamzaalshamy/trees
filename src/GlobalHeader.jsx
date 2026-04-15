import { useNavigate } from "react-router-dom";
import { C } from "./theme";

/**
 * GlobalHeader — persistent top bar used on every page.
 *
 * Props:
 *   right   – ReactNode on the right side of the main bar (Upload CSV, etc.)
 *   infoBar – ReactNode rendered as a second row below the main bar (algorithm pages)
 */
export default function GlobalHeader({ right, infoBar }) {
  const navigate = useNavigate();

  return (
    <div
      style={{
        background: C.bg,
        boxShadow: "0 1px 0 rgba(255,255,255,0.04), 0 4px 20px rgba(0,0,0,0.4)",
        flexShrink: 0,
        position: "relative",
        zIndex: 10,
      }}
    >
      {/* ── Row 1: Logo + right actions ───────────────────────────────────── */}
      <div
        style={{
          height: 52,
          display: "flex",
          alignItems: "center",
          padding: "0 24px",
          gap: 16,
        }}
      >
        {/* Logo — always navigates home */}
        <button
          onClick={() => navigate("/")}
          title="Back to home"
          style={{
            background: "none", border: "none", cursor: "pointer",
            padding: 0, lineHeight: 1, flexShrink: 0,
          }}
        >
          <span
            style={{
              fontWeight: 900, fontSize: 20, letterSpacing: "-0.5px",
              background: `linear-gradient(135deg, ${C.accent}, ${C.green})`,
              WebkitBackgroundClip: "text", WebkitTextFillColor: "transparent",
              backgroundClip: "text",
            }}
          >
            Trees
          </span>
        </button>

        <div style={{ flex: 1 }} />

        {/* Right slot — About link + page-specific content */}
        <div style={{ display: "flex", alignItems: "center", gap: 16 }}>
          <button
            onClick={() => navigate("/about")}
            style={{
              background: "none", border: "none", cursor: "pointer",
              padding: 0, color: C.dim, fontSize: 13, fontWeight: 500,
              transition: "color 0.15s", fontFamily: "inherit",
            }}
            onMouseEnter={(e) => { e.currentTarget.style.color = C.text; }}
            onMouseLeave={(e) => { e.currentTarget.style.color = C.dim; }}
          >
            About
          </button>
          {right}
        </div>
      </div>

      {/* ── Row 2: Algorithm + dataset info bar (algorithm pages only) ────── */}
      {infoBar && (
        <div
          style={{
            borderTop: `1px solid rgba(255,255,255,0.05)`,
            padding: "12px 24px",
            display: "flex",
            alignItems: "center",
            justifyContent: "space-between",
            gap: 24,
          }}
        >
          {infoBar}
        </div>
      )}
    </div>
  );
}
