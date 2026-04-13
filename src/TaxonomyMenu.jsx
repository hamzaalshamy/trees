import { useState, useEffect, useRef, useCallback } from "react";
import { useNavigate } from "react-router-dom";
import { C } from "./theme";
import GlobalHeader from "./GlobalHeader";

// ─── Constants ─────────────────────────────────────────────────────────────────
const VBW = 1060;
const VBH = 510;

const POS = {
  "Decision Tree":     { cx: 530, cy: 80 },
  "Bagging":           { cx: 200, cy: 250 },
  "Boosting":          { cx: 860, cy: 250 },
  "Random Forest":     { cx: 200, cy: 430 },
  "AdaBoost":          { cx: 710, cy: 430 },
  "Gradient Boosting": { cx: 1010, cy: 430 },
};

const NODE_INFO = {
  "Decision Tree": {
    desc: [
      "Learns a sequence of if/else splits directly from data.",
      "At each node: finds the feature + threshold minimizing Gini impurity.",
      "One tree — fully interpretable, the foundation of all ensemble methods.",
    ],
    route: "/decision-tree",
    color: C.accent,
    filterId: "glow-amber",
    sublabel: "single tree",
  },
  "Bagging": {
    desc: [
      "Bootstrap Aggregating — trains many trees independently in parallel.",
      "Each tree is fit on a different bootstrap resample of the training data.",
      "Averages predictions across trees, reducing variance without increasing bias.",
    ],
    route: "/bagging",
    color: C.green,
    filterId: "glow-green",
    sublabel: "bootstrap ensemble",
  },
  "Random Forest": {
    desc: [
      "Extends Bagging by also randomizing the feature set at each split.",
      "Each split considers only √n candidate features — decorrelates the trees.",
      "High accuracy, robust to overfitting, handles mixed feature types well.",
    ],
    route: "/random-forest",
    color: C.green,
    filterId: "glow-green",
    sublabel: "+ feature subsets",
  },
};

const GRAY_NODES = ["Boosting", "AdaBoost", "Gradient Boosting"];

// Edge definitions: { from, to, label, grayed, delay }
const EDGES = [
  { from: "Decision Tree", to: "Bagging",           label: "+ bootstrap",         grayed: false, delay: 0.25 },
  { from: "Decision Tree", to: "Boosting",           label: "+ sequential fitting", grayed: true,  delay: 0.25 },
  { from: "Bagging",       to: "Random Forest",      label: "+ feature subsampling", grayed: false, delay: 0.65 },
  { from: "Boosting",      to: "AdaBoost",            label: null,                 grayed: true,  delay: 0.65 },
  { from: "Boosting",      to: "Gradient Boosting",   label: null,                 grayed: true,  delay: 0.65 },
];

const NODE_DELAYS = {
  "Decision Tree":     0.05,
  "Bagging":           0.45,
  "Boosting":          0.45,
  "Random Forest":     0.85,
  "AdaBoost":          0.85,
  "Gradient Boosting": 0.85,
};

// ─── Sub-components ─────────────────────────────────────────────────────────────
function AnimEdge({ x1, y1, x2, y2, label, grayed, delay }) {
  const strokeColor = grayed ? C.dimmer : C.edge;
  const labelColor  = grayed ? C.dimmer : C.dim;
  const len = Math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2);
  const mx = (x1 + x2) / 2, my = (y1 + y2) / 2;
  const dx = x2 - x1, dy = y2 - y1;
  const nx = -dy / len, ny = dx / len;
  const labelX = mx + nx * 14, labelY = my + ny * 14;

  return (
    <g opacity={grayed ? 0.35 : 0.85}>
      {grayed ? (
        <path
          d={`M${x1},${y1} L${x2},${y2}`}
          stroke={strokeColor}
          strokeWidth={1.5}
          strokeDasharray="5 3"
          fill="none"
          style={{ opacity: 0, animation: `taxFadeIn 0.4s ease ${delay}s forwards` }}
        />
      ) : (
        <path
          d={`M${x1},${y1} L${x2},${y2}`}
          stroke={strokeColor}
          strokeWidth={1.8}
          fill="none"
          style={{
            strokeDasharray: len,
            strokeDashoffset: len,
            animation: `taxDrawEdge 0.55s cubic-bezier(0.4,0,0.2,1) ${delay}s forwards`,
          }}
        />
      )}
      {label && (
        <text
          x={labelX} y={labelY}
          textAnchor="middle"
          fill={labelColor}
          fontSize={9.5}
          fontFamily="'JetBrains Mono',monospace"
          style={{ opacity: 0, animation: `taxFadeIn 0.3s ease ${delay + 0.35}s forwards` }}
        >
          {label}
        </text>
      )}
    </g>
  );
}

function ActiveNode({ label, onHover, onHoverEnd, onClick, animDelay }) {
  const { cx, cy } = POS[label];
  const { color, filterId, sublabel } = NODE_INFO[label];
  const [localHovered, setLocalHovered] = useState(false);
  const w = 170, h = 52;

  return (
    <g
      onClick={onClick}
      onMouseEnter={(e) => { setLocalHovered(true); onHover(label, e.clientX, e.clientY); }}
      onMouseLeave={() => { setLocalHovered(false); onHoverEnd(); }}
      onMouseMove={(e) => onHover(label, e.clientX, e.clientY)}
      style={{
        cursor: "pointer",
        opacity: 0,
        animation: `taxFadeIn 0.4s ease ${animDelay}s forwards`,
      }}
    >
      <rect
        x={cx - w / 2} y={cy - h / 2}
        width={w} height={h} rx={9}
        fill={localHovered ? "#182030" : C.panel}
        stroke={color}
        strokeWidth={localHovered ? 2.4 : 1.8}
        filter={localHovered ? `url(#${filterId})` : undefined}
        style={{ transition: "stroke-width 0.12s, fill 0.12s" }}
      />
      <text
        x={cx} y={cy - 6}
        textAnchor="middle"
        fill={color}
        fontSize={12.5}
        fontFamily="'JetBrains Mono',monospace"
        fontWeight={700}
      >
        {label}
      </text>
      <text
        x={cx} y={cy + 11}
        textAnchor="middle"
        fill={C.dim}
        fontSize={9}
        fontFamily="'JetBrains Mono',monospace"
      >
        {sublabel}
      </text>
    </g>
  );
}

function GrayNode({ label, onComingSoon, animDelay }) {
  const { cx, cy } = POS[label];
  const [localHovered, setLocalHovered] = useState(false);
  const w = 154, h = 44;
  const isLong = label === "Gradient Boosting";
  const nodeW = isLong ? 172 : w;

  return (
    <g
      onClick={() => onComingSoon(label)}
      onMouseEnter={() => setLocalHovered(true)}
      onMouseLeave={() => setLocalHovered(false)}
      style={{
        cursor: "pointer",
        opacity: 0,
        animation: `taxFadeIn 0.4s ease ${animDelay}s forwards`,
      }}
    >
      <rect
        x={cx - nodeW / 2} y={cy - h / 2}
        width={nodeW} height={h} rx={8}
        fill={C.panel}
        stroke={C.dimmer}
        strokeWidth={1.3}
        strokeDasharray="5 3"
        opacity={0.55}
        style={{ transition: "opacity 0.12s" }}
      />
      <text
        x={cx} y={cy + 4.5}
        textAnchor="middle"
        fill={localHovered ? C.dim : C.dimmer}
        fontSize={11}
        fontFamily="'JetBrains Mono',monospace"
        opacity={0.65}
        style={{ transition: "fill 0.12s" }}
      >
        {label}
      </text>
    </g>
  );
}

// ─── How It Works icons ─────────────────────────────────────────────────────────
function IconPick() {
  return (
    <svg width="28" height="28" viewBox="0 0 28 28" fill="none">
      <circle cx="14" cy="14" r="13" stroke={C.accent} strokeWidth="1.5" opacity="0.5" />
      <path d="M10 14 L14 10 L18 14" stroke={C.accent} strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
      <path d="M14 10 L14 19" stroke={C.accent} strokeWidth="1.5" strokeLinecap="round" />
    </svg>
  );
}
function IconWatch() {
  return (
    <svg width="28" height="28" viewBox="0 0 28 28" fill="none">
      <circle cx="14" cy="14" r="13" stroke={C.green} strokeWidth="1.5" opacity="0.5" />
      <circle cx="14" cy="10" r="2" fill={C.green} opacity="0.8" />
      <circle cx="9"  cy="18" r="2" fill={C.green} opacity="0.8" />
      <circle cx="19" cy="18" r="2" fill={C.green} opacity="0.8" />
      <line x1="14" y1="12" x2="10" y2="16" stroke={C.green} strokeWidth="1.2" opacity="0.6" />
      <line x1="14" y1="12" x2="18" y2="16" stroke={C.green} strokeWidth="1.2" opacity="0.6" />
    </svg>
  );
}
function IconUpload() {
  return (
    <svg width="28" height="28" viewBox="0 0 28 28" fill="none">
      <circle cx="14" cy="14" r="13" stroke={C.blue} strokeWidth="1.5" opacity="0.5" />
      <rect x="9" y="14" width="10" height="7" rx="1.5" stroke={C.blue} strokeWidth="1.4" opacity="0.7" />
      <path d="M14 13 L14 7 M11 10 L14 7 L17 10" stroke={C.blue} strokeWidth="1.4" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}

// ─── Main component ─────────────────────────────────────────────────────────────
export default function TaxonomyMenu() {
  const navigate = useNavigate();
  const [tooltip, setTooltip]         = useState(null); // { label, x, y } screen coords
  const [comingSoon, setComingSoon]   = useState(null); // label of clicked gray node
  const [notifyEmail, setNotifyEmail] = useState("");
  const [notifyDone, setNotifyDone]   = useState(false);
  const [dragOver, setDragOver]       = useState(false);
  const fileInputRef = useRef(null);

  // Inject CSS keyframes
  useEffect(() => {
    const id = "taxonomy-keyframes";
    if (document.getElementById(id)) return;
    const style = document.createElement("style");
    style.id = id;
    style.textContent = `
      @keyframes taxDrawEdge { to { stroke-dashoffset: 0; } }
      @keyframes taxFadeIn   { from { opacity: 0; } to { opacity: 1; } }
      @keyframes taxGradientShift {
        0%   { background-position: 0%   50%; }
        50%  { background-position: 100% 50%; }
        100% { background-position: 0%   50%; }
      }
      @keyframes taxDropPulse {
        0%, 100% { border-color: ${C.blue}66; }
        50%       { border-color: ${C.blue}cc; }
      }
    `;
    document.head.appendChild(style);
    return () => document.getElementById(id)?.remove();
  }, []);

  // CSV file handling — navigate to /random-forest and pass the raw CSV content via state
  const handleFile = useCallback((file) => {
    if (!file || !file.name.toLowerCase().endsWith(".csv")) return;
    const reader = new FileReader();
    reader.onload = (e) => {
      navigate("/random-forest", {
        state: { pendingCSV: { name: file.name, content: e.target.result } },
      });
    };
    reader.readAsText(file);
  }, [navigate]);

  const handleDrop = useCallback((e) => {
    e.preventDefault();
    setDragOver(false);
    handleFile(e.dataTransfer.files[0]);
  }, [handleFile]);

  const handleHover = useCallback((label, x, y) => {
    setTooltip({ label, x, y });
  }, []);

  const handleHoverEnd = useCallback(() => {
    setTooltip(null);
  }, []);

  // Close coming-soon on outside click
  useEffect(() => {
    if (!comingSoon) return;
    const handler = (e) => {
      if (!e.target.closest("#coming-soon-popup")) {
        setComingSoon(null);
        setNotifyEmail("");
        setNotifyDone(false);
      }
    };
    document.addEventListener("mousedown", handler);
    return () => document.removeEventListener("mousedown", handler);
  }, [comingSoon]);

  return (
    <div
      style={{
        minHeight: "100vh",
        background: C.bg,
        backgroundImage: `radial-gradient(circle, rgba(255,255,255,0.038) 1px, transparent 1px)`,
        backgroundSize: "28px 28px",
        color: C.text,
        fontFamily: "'JetBrains Mono','Fira Code',monospace",
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        padding: "0 0 0",
      }}
    >
      {/* ── Global Header ───────────────────────────────────────────────────── */}
      <div style={{ width: "100%", flexShrink: 0 }}>
        <GlobalHeader />
      </div>

      {/* ── Hero ────────────────────────────────────────────────────────────── */}
      <div style={{ padding: "32px 24px 0", textAlign: "center", maxWidth: 640 }}>
        <h1
          style={{
            fontSize: "clamp(64px, 9vw, 96px)",
            fontWeight: 900,
            margin: "0 0 12px",
            letterSpacing: "-2px",
            lineHeight: 1,
            background: `linear-gradient(135deg, ${C.accent}, ${C.green}, ${C.accent})`,
            backgroundSize: "200% 200%",
            WebkitBackgroundClip: "text",
            WebkitTextFillColor: "transparent",
            backgroundClip: "text",
            animation: "taxGradientShift 5s ease infinite",
          }}
        >
          Trees
        </h1>
        <p style={{ fontSize: 13, color: C.dim, margin: "0 0 10px", letterSpacing: "0.03em" }}>
          Interactive visualizer for tree-based ML algorithms
        </p>
        <p style={{ fontSize: 14, color: C.text, margin: 0, lineHeight: 1.6, opacity: 0.85 }}>
          Watch machine learning algorithms think, step by step.
          <br />
          <span style={{ color: C.dim }}>Bring your own data.</span>
        </p>
      </div>

      {/* ── Taxonomy SVG ────────────────────────────────────────────────────── */}
      <div style={{ marginTop: 32, position: "relative", width: `min(${VBW}px, 96vw)` }}>
        <svg
          viewBox={`0 0 ${VBW} ${VBH}`}
          style={{ width: "100%", overflow: "visible", display: "block" }}
        >
          <defs>
            <filter id="glow-amber" x="-60%" y="-60%" width="220%" height="220%">
              <feDropShadow dx="0" dy="0" stdDeviation="6" floodColor={C.accent} floodOpacity="0.5" />
            </filter>
            <filter id="glow-green" x="-60%" y="-60%" width="220%" height="220%">
              <feDropShadow dx="0" dy="0" stdDeviation="6" floodColor={C.green} floodOpacity="0.5" />
            </filter>
            <style>{`
              @keyframes taxDrawEdge { to { stroke-dashoffset: 0; } }
              @keyframes taxFadeIn   { from { opacity: 0; } to { opacity: 1; } }
            `}</style>
          </defs>

          {/* Edges */}
          {EDGES.map((e, i) => {
            const { cx: x1, cy: y1 } = POS[e.from];
            const { cx: x2, cy: y2 } = POS[e.to];
            // connect from bottom of source to top of dest (approx)
            const srcH = Object.keys(NODE_INFO).includes(e.from) ? 52 : 44;
            const dstH = Object.keys(NODE_INFO).includes(e.to) ? 52 : 44;
            return (
              <AnimEdge
                key={i}
                x1={x1} y1={y1 + srcH / 2}
                x2={x2} y2={y2 - dstH / 2}
                label={e.label} grayed={e.grayed} delay={e.delay}
              />
            );
          })}

          {/* Active nodes */}
          {Object.keys(NODE_INFO).map((label) => (
            <ActiveNode
              key={label}
              label={label}
              animDelay={NODE_DELAYS[label]}
              onHover={handleHover}
              onHoverEnd={handleHoverEnd}
              onClick={() => navigate(NODE_INFO[label].route)}
            />
          ))}

          {/* Gray nodes */}
          {GRAY_NODES.map((label) => (
            <GrayNode
              key={label}
              label={label}
              animDelay={NODE_DELAYS[label]}
              onComingSoon={setComingSoon}
            />
          ))}
        </svg>

        {/* Hover tooltip */}
        {tooltip && NODE_INFO[tooltip.label] && (
          <div
            style={{
              position: "fixed",
              left: Math.min(tooltip.x + 18, (typeof window !== "undefined" ? window.innerWidth : 1200) - 260),
              top: Math.max(tooltip.y - 110, 8),
              width: 242,
              background: "#141b2d",
              borderRadius: 12,
              padding: "13px 15px",
              pointerEvents: "none",
              zIndex: 100,
              boxShadow: `0 12px 40px rgba(0,0,0,0.7), inset 0 0 0 1px rgba(255,255,255,0.08)`,
            }}
          >
            <div style={{ fontSize: 11, fontWeight: 700, color: NODE_INFO[tooltip.label].color, marginBottom: 8, fontFamily: "'JetBrains Mono',monospace" }}>
              {tooltip.label}
            </div>
            {NODE_INFO[tooltip.label].desc.map((line, i) => (
              <div key={i} style={{ fontSize: 10, color: C.dim, lineHeight: 1.65, fontFamily: "'JetBrains Mono',monospace", marginBottom: i < 2 ? 4 : 0 }}>
                {line}
              </div>
            ))}
            <div style={{ marginTop: 10, fontSize: 10, color: NODE_INFO[tooltip.label].color, fontFamily: "'JetBrains Mono',monospace", opacity: 0.8 }}>
              Click to open →
            </div>
          </div>
        )}
      </div>

      <p style={{ fontSize: 10.5, color: C.dimmer, marginTop: 12, marginBottom: 0, letterSpacing: "0.04em" }}>
        Click a highlighted node to open its interactive visualizer · Hover for details
      </p>

      {/* ── CSV Drop zone ────────────────────────────────────────────────────── */}
      <div
        onDrop={handleDrop}
        onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
        onDragLeave={() => setDragOver(false)}
        onClick={() => fileInputRef.current?.click()}
        style={{
          marginTop: 32,
          width: "min(520px, 90vw)",
          border: `1.5px dashed ${dragOver ? C.blue : "rgba(255,255,255,0.1)"}`,
          borderRadius: 14,
          padding: "20px 28px",
          display: "flex",
          alignItems: "center",
          gap: 18,
          cursor: "pointer",
          background: dragOver ? "#0d1829" : "transparent",
          transition: "border-color 0.2s, background 0.2s",
          animation: dragOver ? "taxDropPulse 0.8s ease infinite" : undefined,
        }}
      >
        <div style={{ flexShrink: 0 }}>
          <svg width="32" height="32" viewBox="0 0 32 32" fill="none">
            <rect x="1" y="1" width="30" height="30" rx="7" stroke={dragOver ? C.blue : C.dim} strokeWidth="1.3" opacity={dragOver ? 0.9 : 0.5} />
            <path d="M16 22 L16 11 M12 15 L16 11 L20 15" stroke={dragOver ? C.blue : C.dim} strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" opacity={dragOver ? 0.9 : 0.6} />
            <path d="M10 24 L22 24" stroke={dragOver ? C.blue : C.dim} strokeWidth="1.5" strokeLinecap="round" opacity={dragOver ? 0.9 : 0.5} />
          </svg>
        </div>
        <div>
          <div style={{ fontSize: 12, color: dragOver ? C.text : C.dim }}>
            {dragOver ? "Drop to load dataset" : "Or drop a CSV to visualize your own data"}
          </div>
          <div style={{ fontSize: 10, color: C.dimmer, marginTop: 5 }}>
            <span style={{ marginRight: 10 }}>
              <span style={{ color: dragOver ? C.blue : C.dim, textDecoration: "underline", cursor: "pointer" }}>
                Browse file
              </span>
              {" "}· .csv files only
            </span>
            <span style={{ opacity: 0.7 }}>🔒 Your data stays in your browser.</span>
          </div>
        </div>
        <input
          ref={fileInputRef}
          type="file"
          accept=".csv"
          style={{ display: "none" }}
          onChange={(e) => handleFile(e.target.files[0])}
          onClick={(e) => e.stopPropagation()}
        />
      </div>

      {/* ── How it works ─────────────────────────────────────────────────────── */}
      <div style={{ marginTop: 64, marginBottom: 64, width: "min(760px, 90vw)" }}>
        <div style={{ textAlign: "center", marginBottom: 32 }}>
          <span style={{ fontSize: 10, color: C.dimmer, letterSpacing: "0.06em" }}>
            How it works
          </span>
        </div>
        <div style={{ display: "flex", gap: 20, flexWrap: "wrap", justifyContent: "center" }}>
          {[
            {
              icon: <IconPick />,
              step: "01",
              title: "Pick an algorithm",
              body: "Choose Decision Tree, Bagging, or Random Forest from the diagram. Each mode locks in the right hyperparameters automatically.",
              color: C.accent,
            },
            {
              icon: <IconWatch />,
              step: "02",
              title: "Watch it build",
              body: "Step through the tree construction one node at a time. The calculations panel shows the exact Gini impurity math at every split.",
              color: C.green,
            },
            {
              icon: <IconUpload />,
              step: "03",
              title: "Try your own data",
              body: "Drop any CSV — the visualizer parses it, lets you pick the target column, handles missing values, and rebuilds the forest instantly.",
              color: C.blue,
            },
          ].map(({ icon, step, title, body, color }) => (
            <div
              key={step}
              style={{
                flex: "1 1 200px",
                maxWidth: 230,
                background: C.panel,
                borderRadius: 16,
                padding: "22px 22px",
                display: "flex",
                flexDirection: "column",
                gap: 12,
                boxShadow: "0 4px 24px rgba(0,0,0,0.4), inset 0 0 0 1px rgba(255,255,255,0.05)",
              }}
            >
              <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
                {icon}
                <span style={{ fontSize: 9, color: C.dimmer, fontWeight: 400 }}>
                  Step {step}
                </span>
              </div>
              <div style={{ fontSize: 12, fontWeight: 700, color }}>
                {title}
              </div>
              <div style={{ fontSize: 10.5, color: C.dim, lineHeight: 1.7 }}>
                {body}
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* ── Footer ───────────────────────────────────────────────────────────── */}
      <footer
        style={{
          boxShadow: "0 -1px 0 rgba(255,255,255,0.04)",
          width: "100%",
          padding: "18px 24px",
          display: "flex",
          justifyContent: "center",
          alignItems: "center",
          gap: 6,
          fontSize: 11,
          color: C.dimmer,
        }}
      >
        Built by{" "}
        <a
          href="https://github.com/hamzaalshamy"
          target="_blank"
          rel="noopener noreferrer"
          style={{ color: C.dim, textDecoration: "none", borderBottom: `1px solid ${C.border}` }}
          onMouseEnter={(e) => { e.target.style.color = C.text; e.target.style.borderBottomColor = C.dim; }}
          onMouseLeave={(e) => { e.target.style.color = C.dim; e.target.style.borderBottomColor = C.border; }}
        >
          Hamza Alshamy
        </a>
        <span style={{ marginLeft: 8, opacity: 0.5 }}>·</span>
        <span style={{ marginLeft: 8, opacity: 0.5 }}>Open source</span>
      </footer>

      {/* ── Coming soon popup ────────────────────────────────────────────────── */}
      {comingSoon && (
        <div
          style={{
            position: "fixed",
            inset: 0,
            background: "rgba(0,0,0,0.55)",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            zIndex: 200,
            backdropFilter: "blur(4px)",
          }}
          onMouseDown={(e) => {
            if (e.target === e.currentTarget) {
              setComingSoon(null);
              setNotifyEmail("");
              setNotifyDone(false);
            }
          }}
        >
          <div
            id="coming-soon-popup"
            style={{
              background: "#141b2d",
              borderRadius: 16,
              padding: "28px 32px",
              width: "min(380px, 90vw)",
              boxShadow: "0 24px 64px rgba(0,0,0,0.7), inset 0 0 0 1px rgba(255,255,255,0.08)",
              fontFamily: "'JetBrains Mono',monospace",
            }}
          >
            <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 6 }}>
              <div style={{ fontSize: 13, fontWeight: 700, color: C.text }}>
                Coming soon
              </div>
              <button
                onClick={() => { setComingSoon(null); setNotifyEmail(""); setNotifyDone(false); }}
                style={{ background: "none", border: "none", color: C.dimmer, cursor: "pointer", fontSize: 16, padding: "2px 6px", lineHeight: 1 }}
              >
                ×
              </button>
            </div>
            <div style={{ fontSize: 11, color: C.dim, marginBottom: 20, lineHeight: 1.6 }}>
              <span style={{ color: C.accent, fontWeight: 700 }}>{comingSoon}</span>
              {" "}is not yet interactive. Leave your email and we'll notify you when it launches.
            </div>
            {notifyDone ? (
              <div style={{
                background: `${C.green}18`,
                border: `1px solid ${C.green}44`,
                borderRadius: 8,
                padding: "12px 16px",
                fontSize: 11,
                color: C.green,
                textAlign: "center",
              }}>
                Thanks! We'll notify you when {comingSoon} is ready.
              </div>
            ) : (
              <div style={{ display: "flex", gap: 8 }}>
                <input
                  type="email"
                  placeholder="your@email.com"
                  value={notifyEmail}
                  onChange={(e) => setNotifyEmail(e.target.value)}
                  onKeyDown={(e) => e.key === "Enter" && notifyEmail && setNotifyDone(true)}
                  style={{
                    flex: 1,
                    background: "rgba(255,255,255,0.05)",
                    border: "none",
                    borderRadius: 9,
                    color: C.text,
                    fontSize: 11,
                    padding: "8px 12px",
                    fontFamily: "'JetBrains Mono',monospace",
                    outline: "none",
                    boxShadow: "inset 0 0 0 1px rgba(255,255,255,0.1)",
                  }}
                />
                <button
                  onClick={() => notifyEmail && setNotifyDone(true)}
                  style={{
                    background: notifyEmail ? C.accent : "rgba(255,255,255,0.08)",
                    border: "none",
                    borderRadius: 9,
                    color: notifyEmail ? "#000" : C.dimmer,
                    fontSize: 11,
                    fontWeight: 700,
                    cursor: notifyEmail ? "pointer" : "default",
                    padding: "8px 14px",
                    fontFamily: "'JetBrains Mono',monospace",
                    transition: "background 0.15s ease-out, color 0.15s ease-out",
                  }}
                >
                  Notify me
                </button>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
