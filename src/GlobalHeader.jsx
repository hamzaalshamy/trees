import { useNavigate } from "react-router-dom";
import { C } from "./theme";

/**
 * GlobalHeader — persistent top bar used on every page.
 *
 * Props:
 *   breadcrumb  – array of { label, route? } items, e.g. [{ label:'Trees', route:'/' }, { label:'Random Forest' }]
 *                 rendered as "Trees → Random Forest" with the routed items clickable
 *   description – short algorithm description shown below the breadcrumb row (string or ReactNode)
 *   center      – ReactNode rendered in the middle of the main bar (dataset badge on algorithm pages)
 *   right       – ReactNode rendered on the right side (Upload CSV button, reset button, etc.)
 */
export default function GlobalHeader({ breadcrumb, description, center, right }) {
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
      {/* ── Main bar ──────────────────────────────────────────────────────── */}
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
            background: "none",
            border: "none",
            cursor: "pointer",
            padding: 0,
            lineHeight: 1,
            flexShrink: 0,
          }}
        >
          <span
            style={{
              fontFamily: "'JetBrains Mono','Fira Code',monospace",
              fontWeight: 900,
              fontSize: 20,
              letterSpacing: "-0.5px",
              background: `linear-gradient(135deg, ${C.accent}, ${C.green})`,
              WebkitBackgroundClip: "text",
              WebkitTextFillColor: "transparent",
              backgroundClip: "text",
            }}
          >
            Trees
          </span>
        </button>

        {/* Center slot — dataset badge on algorithm pages */}
        {center && (
          <div style={{ flex: 1, display: "flex", justifyContent: "center" }}>
            {center}
          </div>
        )}

        {/* Spacer when no center slot */}
        {!center && <div style={{ flex: 1 }} />}

        {/* Right slot — About link + page-specific content (Upload CSV, reset, etc.) */}
        <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
          <button
            onClick={() => navigate("/about")}
            style={{
              background: "none",
              border: "none",
              cursor: "pointer",
              padding: 0,
              color: C.text,
              fontSize: 14,
              fontWeight: 600,
              fontFamily: "'JetBrains Mono','Fira Code',monospace",
              transition: "color 0.15s",
            }}
            onMouseEnter={(e) => { e.currentTarget.style.color = C.accent; }}
            onMouseLeave={(e) => { e.currentTarget.style.color = C.text; }}
          >
            About
          </button>
          {right && (
            <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
              {right}
            </div>
          )}
        </div>
      </div>

      {/* ── Breadcrumb + description row (algorithm pages only) ───────────── */}
      {(breadcrumb || description) && (
        <div
          style={{
            height: 26,
            display: "flex",
            alignItems: "center",
            padding: "0 24px",
            gap: 12,
            borderTop: `1px solid rgba(255,255,255,0.04)`,
          }}
        >
          {/* Breadcrumb */}
          {breadcrumb && (
            <div
              style={{
                display: "flex",
                alignItems: "center",
                gap: 5,
                fontSize: 10,
                fontFamily: "'JetBrains Mono',monospace",
                flexShrink: 0,
              }}
            >
              {breadcrumb.map((item, i) => (
                <span key={i} style={{ display: "flex", alignItems: "center", gap: 5 }}>
                  {i > 0 && (
                    <span style={{ color: C.dimmer, fontSize: 9 }}>›</span>
                  )}
                  {item.route ? (
                    <button
                      onClick={() => navigate(item.route)}
                      style={{
                        background: "none",
                        border: "none",
                        cursor: "pointer",
                        padding: 0,
                        color: C.dim,
                        fontSize: 10,
                        fontFamily: "inherit",
                        textDecoration: "none",
                      }}
                      onMouseEnter={(e) => { e.target.style.color = C.text; }}
                      onMouseLeave={(e) => { e.target.style.color = C.dim; }}
                    >
                      {item.label}
                    </button>
                  ) : (
                    <span style={{ color: C.text, fontWeight: 600 }}>{item.label}</span>
                  )}
                </span>
              ))}
            </div>
          )}

          {/* Separator */}
          {breadcrumb && description && (
            <span style={{ color: C.dimmer, fontSize: 9 }}>·</span>
          )}

          {/* Algorithm description */}
          {description && (
            <span
              style={{
                fontSize: 10,
                color: C.dim,
                fontFamily: "'JetBrains Mono',monospace",
              }}
            >
              {description}
            </span>
          )}
        </div>
      )}
    </div>
  );
}
