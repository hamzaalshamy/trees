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

const COMING_SOON_DESC = {
  "Boosting":           "Boosting — sequential ensemble methods where each tree corrects the previous one's mistakes.",
  "AdaBoost":           "AdaBoost — watch misclassified samples gain weight as each weak learner focuses on harder examples.",
  "Gradient Boosting":  "Gradient Boosting — may be added in the future. Requires visualizing residual fitting, a fundamentally different approach.",
};

// Edge definitions: { from, to, label, grayed, delay }
const EDGES = [
  { from: "Decision Tree", to: "Bagging",           label: "+ bootstrap",          grayed: false, delay: 0.25 },
  { from: "Decision Tree", to: "Boosting",          label: "+ sequential fitting",  grayed: true,  delay: 0.25 },
  { from: "Bagging",       to: "Random Forest",     label: "+ feature subsampling", grayed: false, delay: 0.65 },
  { from: "Boosting",      to: "AdaBoost",           label: null,                   grayed: true,  delay: 0.65 },
  { from: "Boosting",      to: "Gradient Boosting",  label: null,                   grayed: true,  delay: 0.65 },
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
function AnimEdge({ x1, y1, x2, y2, label, grayed, delay, hovered }) {
  const strokeColor = grayed ? C.dimmer : (hovered ? C.dim : C.edge);
  const labelColor  = grayed ? C.dimmer : (hovered ? C.text : C.dim);
  const len = Math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2);
  const mx = (x1 + x2) / 2, my = (y1 + y2) / 2;
  const dx = x2 - x1, dy = y2 - y1;
  const nx = -dy / len, ny = dx / len;
  const labelX = mx + nx * 14, labelY = my + ny * 14;

  return (
    <g opacity={grayed ? 0.35 : (hovered ? 1 : 0.85)} style={{ transition: "opacity 0.2s" }}>
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
          strokeWidth={hovered ? 2.4 : 1.8}
          fill="none"
          style={{
            strokeDasharray: len,
            strokeDashoffset: len,
            animation: `taxDrawEdge 0.55s cubic-bezier(0.4,0,0.2,1) ${delay}s forwards`,
            transition: "stroke-width 0.2s, stroke 0.2s",
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
        transform: localHovered ? "scale(1.05)" : "scale(1)",
        transformOrigin: `${cx}px ${cy}px`,
        transition: "transform 0.18s cubic-bezier(0.34,1.56,0.64,1)",
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
        animation: localHovered
          ? `taxFadeIn 0.4s ease ${animDelay}s forwards, taxShake 0.38s ease`
          : `taxFadeIn 0.4s ease ${animDelay}s forwards`,
        transformOrigin: `${cx}px ${cy}px`,
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
        fill={localHovered ? C.text : C.dim}
        fontSize={11}
        fontFamily="'JetBrains Mono',monospace"
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
      {/* Root node */}
      <circle cx="14" cy="7" r="3.5" stroke={C.accent} strokeWidth="1.4" fill={`${C.accent}18`} />
      {/* Left edge */}
      <line x1="11" y1="9.5" x2="7.5" y2="16.5" stroke={C.accent} strokeWidth="1.2" opacity="0.65" strokeLinecap="round" />
      {/* Right edge */}
      <line x1="17" y1="9.5" x2="20.5" y2="16.5" stroke={C.accent} strokeWidth="1.2" opacity="0.65" strokeLinecap="round" />
      {/* Left leaf */}
      <rect x="3" y="17.5" width="9" height="5.5" rx="2" stroke={C.accent} strokeWidth="1.2" fill={`${C.accent}20`} />
      {/* Right leaf */}
      <rect x="16" y="17.5" width="9" height="5.5" rx="2" stroke={C.accent} strokeWidth="1.2" fill={`${C.accent}20`} />
    </svg>
  );
}
function IconWatch() {
  return (
    <svg width="28" height="28" viewBox="0 0 28 28" fill="none">
      {/* Node box */}
      <rect x="7" y="9" width="14" height="10" rx="2.5" stroke={C.green} strokeWidth="1.4" fill={`${C.green}14`} />
      {/* Interior rule line */}
      <line x1="10" y1="14" x2="18" y2="14" stroke={C.green} strokeWidth="1" opacity="0.45" />
      {/* Left step arrow */}
      <path d="M5 14 L2.5 14 M2.5 14 L4.5 12.2 M2.5 14 L4.5 15.8" stroke={C.green} strokeWidth="1.2" strokeLinecap="round" strokeLinejoin="round" opacity="0.7" />
      {/* Right step arrow */}
      <path d="M23 14 L25.5 14 M25.5 14 L23.5 12.2 M25.5 14 L23.5 15.8" stroke={C.green} strokeWidth="1.2" strokeLinecap="round" strokeLinejoin="round" opacity="0.7" />
    </svg>
  );
}
function IconUpload() {
  return (
    <svg width="28" height="28" viewBox="0 0 28 28" fill="none">
      {/* Document body */}
      <path d="M7 4 L18 4 L22 8 L22 24 L7 24 Z" stroke={C.blue} strokeWidth="1.4" fill={`${C.blue}12`} strokeLinejoin="round" />
      {/* Folded corner */}
      <path d="M18 4 L18 8 L22 8" stroke={C.blue} strokeWidth="1.2" fill="none" opacity="0.6" strokeLinejoin="round" />
      {/* Upward arrow */}
      <path d="M14.5 20 L14.5 13 M12 15 L14.5 13 L17 15" stroke={C.blue} strokeWidth="1.4" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}

// ─── Main component ─────────────────────────────────────────────────────────────
export default function TaxonomyMenu() {
  const navigate = useNavigate();
  const [tooltip, setTooltip]           = useState(null); // { label, x, y } screen coords
  const [comingSoon, setComingSoon]     = useState(null); // label of clicked gray node
  const [dragOver, setDragOver]         = useState(false);
  const [cardsVisible, setCardsVisible] = useState(false);
  const fileInputRef = useRef(null);
  const cardsRef     = useRef(null);

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
      @keyframes taxShake {
        0%, 100% { transform: translateX(0); }
        20%      { transform: translateX(-4px); }
        40%      { transform: translateX(4px); }
        60%      { transform: translateX(-2.5px); }
        80%      { transform: translateX(2.5px); }
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

  // Scroll-in animation for How it Works cards — fires once, then disconnects
  useEffect(() => {
    const el = cardsRef.current;
    if (!el) return;
    const obs = new IntersectionObserver(
      ([entry]) => { if (entry.isIntersecting) { setCardsVisible(true); obs.disconnect(); } },
      { threshold: 0.15 }
    );
    obs.observe(el);
    return () => obs.disconnect();
  }, []);

  // Close coming-soon on outside click
  useEffect(() => {
    if (!comingSoon) return;
    const handler = (e) => {
      if (!e.target.closest("#coming-soon-popup")) setComingSoon(null);
    };
    document.addEventListener("mousedown", handler);
    return () => document.removeEventListener("mousedown", handler);
  }, [comingSoon]);

  return (
    <div
      style={{
        minHeight: "100vh",
        position: "relative",
        background: C.bg,
        backgroundImage: `radial-gradient(circle, rgba(255,255,255,0.038) 1px, transparent 1px)`,
        backgroundSize: "28px 28px",
        color: C.text,
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
        {/* Radial glow centered on Decision Tree node.
            Extended 200px above the container so the gradient fades to transparent
            before hitting the top edge — no hard clip. Center % recalculated
            to keep the focal point on the node (~cy=80 in the 510px-tall SVG). */}
        <div
          style={{
            position: "absolute",
            top: -200, left: 0, right: 0, bottom: 0,
            background: `radial-gradient(ellipse 85% 45% at 50% 39%, ${C.accent}0d 0%, ${C.green}08 45%, transparent 82%)`,
            pointerEvents: "none",
            zIndex: 0,
          }}
        />
        <svg
          viewBox={`0 0 ${VBW} ${VBH}`}
          style={{ width: "100%", overflow: "visible", display: "block", position: "relative", zIndex: 1 }}
        >
          <defs>
            <filter id="glow-amber" x="-80%" y="-80%" width="260%" height="260%">
              <feDropShadow dx="0" dy="0" stdDeviation="9" floodColor={C.accent} floodOpacity="0.65" />
            </filter>
            <filter id="glow-green" x="-80%" y="-80%" width="260%" height="260%">
              <feDropShadow dx="0" dy="0" stdDeviation="9" floodColor={C.green} floodOpacity="0.65" />
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
            const srcH = Object.keys(NODE_INFO).includes(e.from) ? 52 : 44;
            const dstH = Object.keys(NODE_INFO).includes(e.to) ? 52 : 44;
            return (
              <AnimEdge
                key={i}
                x1={x1} y1={y1 + srcH / 2}
                x2={x2} y2={y2 - dstH / 2}
                label={e.label} grayed={e.grayed} delay={e.delay}
                hovered={!e.grayed && (e.from === tooltip?.label || e.to === tooltip?.label)}
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
            <div style={{ fontSize: 11, fontWeight: 700, color: NODE_INFO[tooltip.label].color, marginBottom: 8 }}>
              {tooltip.label}
            </div>
            {NODE_INFO[tooltip.label].desc.map((line, i) => (
              <div key={i} style={{ fontSize: 10, color: C.text, opacity: 0.7, lineHeight: 1.65, marginBottom: i < 2 ? 4 : 0 }}>
                {line}
              </div>
            ))}
            <div style={{ marginTop: 10, fontSize: 10, color: NODE_INFO[tooltip.label].color, opacity: 0.8 }}>
              Click to open →
            </div>
          </div>
        )}
      </div>

      <p style={{ fontSize: 10.5, color: C.dim, marginTop: 12, marginBottom: 0, letterSpacing: "0.04em" }}>
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
          border: `1.5px solid ${dragOver ? C.blue : "rgba(255,255,255,0.1)"}`,
          borderRadius: 14,
          padding: "20px 28px",
          display: "flex",
          alignItems: "center",
          gap: 18,
          cursor: "pointer",
          background: dragOver ? "#0d1829" : "rgba(255,255,255,0.015)",
          transition: "border-color 0.2s, background 0.2s",
          boxShadow: dragOver ? `0 0 0 3px ${C.blue}22` : "none",
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
            {dragOver ? "Drop to load dataset" : "Have your own dataset? Drop a CSV and watch your data come alive."}
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
          <span style={{ fontSize: 12, color: C.text, fontWeight: 600, letterSpacing: "0.04em" }}>
            How it works
          </span>
        </div>
        <div ref={cardsRef} style={{ display: "flex", gap: 20, flexWrap: "wrap", justifyContent: "center" }}>
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
          ].map(({ icon, step, title, body, color }, idx) => (
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
                opacity: cardsVisible ? 1 : 0,
                transform: cardsVisible ? "translateY(0)" : "translateY(20px)",
                transition: `opacity 0.5s ease ${idx * 0.1}s, transform 0.5s ease ${idx * 0.1}s`,
              }}
            >
              <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
                {icon}
                <span style={{ fontSize: 9, color: C.dim, fontWeight: 400 }}>
                  Step {step}
                </span>
              </div>
              <div style={{ fontSize: 12, fontWeight: 700, color }}>
                {title}
              </div>
              <div style={{ fontSize: 10.5, color: C.text, lineHeight: 1.7, opacity: 0.75 }}>
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
          flexDirection: "column",
          justifyContent: "center",
          alignItems: "center",
          gap: 6,
          fontSize: 11,
          color: C.dimmer,
        }}
      >
        <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
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
          <span style={{ opacity: 0.5 }}>·</span>
          <span style={{ opacity: 0.5 }}>Open source</span>
        </div>
        <a
          href="https://hamzaalshamy.github.io"
          target="_blank"
          rel="noopener noreferrer"
          style={{ color: C.dimmer, textDecoration: "none", fontSize: 10.5, opacity: 0.7 }}
          onMouseEnter={(e) => { e.target.style.color = C.dim; e.target.style.opacity = 1; }}
          onMouseLeave={(e) => { e.target.style.color = C.dimmer; e.target.style.opacity = 0.7; }}
        >
          hamzaalshamy.github.io
        </a>
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
          onMouseDown={(e) => { if (e.target === e.currentTarget) setComingSoon(null); }}
        >
          <div
            id="coming-soon-popup"
            style={{
              background: "#141b2d",
              borderRadius: 16,
              padding: "26px 28px",
              width: "min(360px, 90vw)",
              boxShadow: "0 24px 64px rgba(0,0,0,0.7), inset 0 0 0 1px rgba(255,255,255,0.08)",
            }}
          >
            <div style={{ fontSize: 11, fontWeight: 600, color: C.dimmer, marginBottom: 10, letterSpacing: "0.04em", textTransform: "uppercase" }}>
              Coming soon
            </div>
            <div style={{ fontSize: 12, color: C.dim, lineHeight: 1.7, marginBottom: 20 }}>
              {COMING_SOON_DESC[comingSoon]}
            </div>
            <button
              onClick={() => setComingSoon(null)}
              onMouseEnter={e => { e.currentTarget.style.background = "rgba(255,255,255,0.08)"; e.currentTarget.style.color = C.text; }}
              onMouseLeave={e => { e.currentTarget.style.background = "rgba(255,255,255,0.05)"; e.currentTarget.style.color = C.dim; }}
              style={{
                background: "rgba(255,255,255,0.05)",
                border: "none", borderRadius: 8,
                color: C.dim, fontSize: 11, fontWeight: 500,
                cursor: "pointer", padding: "7px 16px",
                transition: "background 0.15s, color 0.15s",
              }}
            >
              Got it
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
