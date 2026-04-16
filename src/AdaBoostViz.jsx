import { useState, useEffect, useRef, useCallback, useMemo } from "react";
import { createPortal } from "react-dom";
import { useLocation } from "react-router-dom";
import Papa from "papaparse";
import { C, MONO } from "./theme";
import GlobalHeader from "./GlobalHeader";
import { heartData, heartMeta } from "./data/heartDisease";
import { musicData, musicMeta } from "./data/musicData";
import { predictAdaTree, adaBoostEnsembleScores } from "./adaBoostAlgorithm";
import AdaBoostWorker from "./adaBoostWorker.js?worker";
import DataModal from "./DataModal";
import { NA_VALS, detectNAs } from "./dataUtils";
import { CLASS_PALETTE, classColor, flattenNodes, LEAF_SPACING, X_PAD, Y_GAP, computeTreeWidth, computePositions, getSamplePath, formatSampleLabel, fmtThresh, PATH_COLOR, Edge, TreeNode } from "./TreeComponents";
import { tooltipPosition } from "./tooltipUtils";

const EMPTY_TS = { visibleIds: [], nodeId: null, phase: 0, stepIdx: -1 };

// ─── Cumulative Error Chart ────────────────────────────────────────────────────
function ErrorChart({ accuracies, completedCount }) {
  const [hovered, setHovered] = useState(null); // index of hovered point
  const W = 220, H = 110, PAD = { t: 10, r: 10, b: 24, l: 36 };
  const iW = W - PAD.l - PAD.r, iH = H - PAD.t - PAD.b;
  const visible = accuracies.slice(0, completedCount).map(a => 100 - (a ?? 100));
  if (!visible.length) return (
    <div style={{ height: H, display: "flex", alignItems: "center", justifyContent: "center" }}>
      <span style={{ fontSize: 9, color: C.dimmer }}>Builds after rounds complete</span>
    </div>
  );
  const maxErr = Math.max(...visible, 0.1);
  const minErr = 0;
  const xScale = n => PAD.l + (n / Math.max(accuracies.length - 1, 1)) * iW;
  const yScale = v => PAD.t + iH - ((v - minErr) / (maxErr - minErr || 1)) * iH;
  const pts = visible.map((v, i) => [xScale(i), yScale(v)]);
  const path = pts.map((p, i) => `${i === 0 ? "M" : "L"}${p[0].toFixed(1)},${p[1].toFixed(1)}`).join(" ");
  const area = [
    `M${pts[0][0].toFixed(1)},${(PAD.t + iH).toFixed(1)}`,
    ...pts.map(p => `L${p[0].toFixed(1)},${p[1].toFixed(1)}`),
    `L${pts[pts.length - 1][0].toFixed(1)},${(PAD.t + iH).toFixed(1)}`,
    "Z",
  ].join(" ");
  const yTicks = [0, maxErr / 2, maxErr].map(v => ({ val: v, y: yScale(v) }));
  const xLabels = [1, Math.ceil(accuracies.length / 2), accuracies.length].filter((v, i, arr) => arr.indexOf(v) === i);
  const hp = hovered !== null ? pts[hovered] : null;

  return (
    <svg width={W} height={H} style={{ overflow: "visible", display: "block" }}>
      <defs>
        <linearGradient id="errGrad" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor={C.orange} stopOpacity="0.28" />
          <stop offset="100%" stopColor={C.orange} stopOpacity="0" />
        </linearGradient>
      </defs>
      {yTicks.map(({ y }, i) => (
        <line key={i} x1={PAD.l} y1={y} x2={PAD.l + iW} y2={y}
          stroke="rgba(255,255,255,0.05)" strokeWidth={1} />
      ))}
      <path d={area} fill="url(#errGrad)" />
      <path d={path} fill="none" stroke={C.orange} strokeWidth={1.8} strokeLinejoin="round" strokeLinecap="round" />
      {/* Invisible hit-area rects between points */}
      {pts.map(([x], i) => (
        <rect key={i}
          x={i === 0 ? PAD.l : (pts[i-1][0] + x) / 2}
          y={PAD.t}
          width={i === pts.length - 1 ? PAD.l + iW - (i === 0 ? PAD.l : (pts[i-1][0] + x) / 2)
            : ((x + pts[i+1][0]) / 2) - (i === 0 ? PAD.l : (pts[i-1][0] + x) / 2)}
          height={iH}
          fill="transparent"
          onMouseEnter={() => setHovered(i)}
          onMouseLeave={() => setHovered(null)}
          style={{ cursor: "crosshair" }}
        />
      ))}
      {pts.map(([x, y], i) => (
        <circle key={i} cx={x} cy={y} r={hovered === i ? 4 : 2.5}
          fill={C.orange} style={{ transition: "r 0.1s", pointerEvents: "none" }} />
      ))}
      {yTicks.map(({ val, y }, i) => (
        <text key={i} x={PAD.l - 5} y={y + 3} textAnchor="end" fill={C.dimmer} fontSize={7.5} fontFamily={MONO}>
          {val.toFixed(0)}%
        </text>
      ))}
      {xLabels.map(v => (
        <text key={v} x={xScale(v - 1)} y={H - 5} textAnchor="middle" fill={C.dimmer} fontSize={7.5} fontFamily={MONO}>
          R{v}
        </text>
      ))}
      <line x1={PAD.l} y1={PAD.t} x2={PAD.l} y2={PAD.t + iH} stroke={C.edge} strokeWidth={1} />
      <line x1={PAD.l} y1={PAD.t + iH} x2={PAD.l + iW} y2={PAD.t + iH} stroke={C.edge} strokeWidth={1} />
      {/* Hover tooltip */}
      {hp && hovered !== null && (() => {
        const tx = hp[0] + (hp[0] > PAD.l + iW * 0.6 ? -58 : 6);
        const ty = Math.max(PAD.t + 2, hp[1] - 18);
        return (
          <g pointerEvents="none">
            <rect x={tx - 4} y={ty - 10} width={56} height={20} rx={4}
              fill="#1a2235" stroke="rgba(255,255,255,0.12)" strokeWidth={0.8} />
            <text x={tx + 24} y={ty + 4} textAnchor="middle" fill={C.orange}
              fontSize={8} fontFamily={MONO} fontWeight={600}>
              R{hovered + 1}: {visible[hovered].toFixed(1)}% err
            </text>
          </g>
        );
      })()}
    </svg>
  );
}

// ─── Weight Bar Chart ──────────────────────────────────────────────────────────
function WeightChart({ weights, sortWeights, classColors, predCorrect, showPredColors, maxH = 120 }) {
  const n = weights?.length ?? 0;
  if (!n) return null;
  const useHistogram = n > 500;
  const maxW = Math.max(...weights, 1e-10);
  const avgW  = 1 / n;
  const avgH  = Math.max(1, (avgW / maxW) * maxH);

  if (useHistogram) {
    const BINS = 30;
    const binW = maxW / BINS;
    const counts = new Array(BINS).fill(0);
    weights.forEach(w => {
      const bi = Math.min(BINS - 1, Math.floor(w / binW));
      counts[bi]++;
    });
    const maxC = Math.max(...counts, 1);
    return (
      <div style={{ position: "relative" }}>
        <div style={{ display: "flex", alignItems: "flex-end", gap: 1, height: maxH }}>
          {counts.map((c, i) => (
            <div key={i} style={{
              flex: 1, height: `${(c / maxC) * maxH}px`,
              background: C.accent, borderRadius: "2px 2px 0 0",
              minHeight: c > 0 ? 2 : 0,
            }} />
          ))}
        </div>
        <div style={{ position: "absolute", bottom: avgH, left: 0, right: 0, height: 1,
          background: "rgba(255,255,255,0.3)", pointerEvents: "none" }}>
          <span style={{ position: "absolute", right: 2, top: -8, fontSize: 7,
            color: "rgba(255,255,255,0.45)", fontFamily: MONO }}>1/n</span>
        </div>
      </div>
    );
  }

  // Sort indices by sortWeights (or weights) descending — decay curve shape
  const base = sortWeights ?? weights;
  const sortedIndices = [...Array(n).keys()].sort((a, b) => base[b] - base[a]);

  return (
    <div style={{ position: "relative" }}>
      <div style={{ display: "flex", alignItems: "flex-end", height: maxH, gap: 0, overflow: "hidden" }}>
        {sortedIndices.map((i) => {
          let bg = classColors[i] ?? C.dim;
          if (showPredColors && predCorrect !== null) {
            bg = predCorrect[i] ? "#10b981" : "#ef4444";
          }
          return (
            <div key={i} style={{
              flex: 1, minWidth: 1,
              height: `${Math.max(2, (weights[i] / maxW) * maxH)}px`,
              background: bg,
              borderRadius: "1px 1px 0 0",
              transition: "height 0.55s ease, background 0.3s ease",
              opacity: 0.85,
            }} />
          );
        })}
      </div>
      {/* Average weight reference line */}
      <div style={{ position: "absolute", bottom: avgH, left: 0, right: 0, height: 1,
        background: "rgba(255,255,255,0.28)", pointerEvents: "none" }}>
        <span style={{ position: "absolute", right: 2, top: -8, fontSize: 7,
          color: "rgba(255,255,255,0.45)", fontFamily: MONO }}>1/n</span>
      </div>
    </div>
  );
}

// ─── Phase label ──────────────────────────────────────────────────────────────
const PHASE_LABELS = [
  { color: C.accent,  text: "Weights",          icon: "⚖" },
  { color: C.accent,  text: "Building tree",    icon: "🌲" },
  { color: C.green,   text: "Predictions",      icon: "✓" },
  { color: C.orange,  text: "Error & α",        icon: "α" },
  { color: C.purple,  text: "Updating weights", icon: "↑↓" },
  { color: C.green,   text: "Round complete",   icon: "✔" },
];

// ─── Main component ────────────────────────────────────────────────────────────
export default function AdaBoostViz() {
  const location = useLocation();

  // ── Dataset ─────────────────────────────────────────────────────────────────
  const [builtinDataset, setBuiltinDataset] = useState("heart");
  const [customDataset,  setCustomDataset]  = useState(null);
  const fileInputRef = useRef(null);
  const [csvModal, setCsvModal] = useState(null);
  const [dragOver, setDragOver] = useState(false);

  const builtinMeta = builtinDataset === "music" ? musicMeta : heartMeta;
  const builtinData = builtinDataset === "music" ? musicData : heartData;
  const activeData      = customDataset?.data        ?? builtinData;
  const activeFeatures  = customDataset?.features    ?? builtinMeta.features;
  const activeTargetCol = customDataset?.targetCol   ?? builtinMeta.targetCol;
  const classLabels     = customDataset?.classLabels ?? builtinMeta.targetLabels;
  const allClasses = useMemo(() => {
    if (Array.isArray(classLabels)) return classLabels;
    if (classLabels && typeof classLabels === "object") return Object.values(classLabels);
    return [...new Set(activeData.map(r => r[activeTargetCol]))].sort();
  }, [classLabels, activeData, activeTargetCol]);

  // ── Hyperparams ─────────────────────────────────────────────────────────────
  const [maxDepth,    setMaxDepth]    = useState(1);
  const [maxDepthStr, setMaxDepthStr] = useState("1");
  const [nRounds,     setNRounds]     = useState(10);
  const [nRoundsStr,  setNRoundsStr]  = useState("10");
  const [speed,       setSpeed]       = useState(1);

  // ── Round state ──────────────────────────────────────────────────────────────
  const [roundData,          setRoundData]          = useState([]);   // pre-built
  const [cumulativeAccuracy, setCumulativeAccuracy] = useState([]);
  const [curRound,           setCurRound]           = useState(0);
  const [completedCount,     setCompletedCount]     = useState(0);    // rounds through phase 5
  const [roundPhase,         setRoundPhase]         = useState(-1);   // -1 = not started
  const [treeStates,         setTreeStates]         = useState({});
  const [growing,            setGrowing]            = useState(false);
  const [buildProgress,      setBuildProgress]      = useState(false);
  const [selectedSampleIdx,  setSelectedSampleIdx]  = useState(null);
  const [lockedTip,          setLockedTip]          = useState(null); // { top, left } | null
  const [resetTooltip,       setResetTooltip]       = useState(null); // { x, y }
  const [weightDisplayMode,  setWeightDisplayMode]  = useState("input"); // "input"|"output"
  const [calcExpanded,       setCalcExpanded]       = useState(true);
  const [calcPanelWidth,     setCalcPanelWidth]     = useState(220);
  const [hintDismissed,      setHintDismissed]      = useState(false);
  const [dsDropOpen,         setDsDropOpen]         = useState(false);
  const [dsDropPos,          setDsDropPos]          = useState(null); // { top, left } for portaled menu
  const [dsOptTooltip,       setDsOptTooltip]       = useState(null); // { top, left, text }
  const [csvError,           setCsvError]           = useState(null); // { type, name, nClasses? }
  const justRebuiltRef = useRef(false);
  const dsDropRef      = useRef(null);
  const dsMenuRef      = useRef(null);

  const growRef      = useRef(false);
  const cancelRef    = useRef(false);
  const workerRef    = useRef(null);
  const liveStepRef   = useRef({ roundIdx: 0, phase: 0, stepIdx: -1 });
  const pausedAtRef   = useRef(null);
  const resetClickRef = useRef(null);

  // Zoom / pan
  const [zoom, setZoom] = useState(1);
  const [pan,  setPan]  = useState({ x: 0, y: 0 });
  const [cursorGrabbing, setCursorGrabbing] = useState(false);
  const isDragging = useRef(false);
  const dragStart  = useRef({ x: 0, y: 0, px: 0, py: 0 });
  const canvasRef  = useRef(null);
  const zoomLive   = useRef(1);
  const panLive    = useRef({ x: 0, y: 0 });
  zoomLive.current = zoom;
  panLive.current  = pan;

  const predSectionRef = useRef(null);
  const [scrollHint,   setScrollHint]   = useState(false);

  // Inject CSS keyframes
  useEffect(() => {
    const id = "ab-keyframes";
    if (document.getElementById(id)) return;
    const s = document.createElement("style");
    s.id = id;
    s.textContent = `
      @keyframes nodeIn { from { opacity: 0; transform: scale(0.94); } to { opacity: 1; transform: scale(1); } }
      @keyframes abFadeIn { from { opacity: 0; } to { opacity: 1; } }
      @keyframes taxFadeIn { from { opacity: 0; } to { opacity: 1; } }
      .ab-ds-drop-menu { position: fixed; z-index: 99999; min-width: 230px;
        background: #1a2235; border: 1px solid rgba(255,255,255,0.12); border-radius: 8px; padding: 4px;
        box-shadow: 0 8px 32px rgba(0,0,0,0.6); }
      .ab-ds-opt { display: flex; align-items: center; gap: 6px; width: 100%; padding: 7px 10px; border-radius: 5px;
        border: none; background: none; font-family: 'Inter', system-ui, sans-serif; font-size: 11px;
        text-align: left; cursor: pointer; white-space: nowrap; color: #e2e8f0; transition: background 0.12s; box-sizing: border-box; }
      .ab-ds-opt:hover:not([data-disabled="true"]) { background: rgba(255,255,255,0.08); }
      .ab-ds-opt[data-disabled="true"] { color: #3d4a5c; cursor: not-allowed; }
      .ab-ds-opt[data-active="true"] { color: #f59e0b; }
      select.ds-pill-ab {
        -webkit-appearance: none; appearance: none;
        background: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.1);
        border-radius: 20px; color: #e2e8f0; font-family: inherit; font-size: 12px;
        font-weight: 500; padding: 5px 28px 5px 12px; cursor: pointer; outline: none;
        transition: box-shadow 0.2s ease, border-color 0.2s ease; min-width: 0;
      }
      select.ds-pill-ab:hover { border-color: rgba(255,255,255,0.2); box-shadow: 0 0 0 3px rgba(245,158,11,0.12); }
      select.ds-pill-ab:focus { border-color: rgba(245,158,11,0.4); box-shadow: 0 0 0 3px rgba(245,158,11,0.12); }
      select.ds-pill-ab option { background: #111827; color: #e2e8f0; }
    `;
    document.head.appendChild(s);
    return () => document.getElementById(id)?.remove();
  }, []);

  // Close dataset dropdown on outside click
  useEffect(() => {
    if (!dsDropOpen) return;
    const handler = (e) => {
      const inTrigger = dsDropRef.current?.contains(e.target);
      const inMenu    = dsMenuRef.current?.contains(e.target);
      if (!inTrigger && !inMenu) setDsDropOpen(false);
    };
    document.addEventListener("mousedown", handler);
    return () => document.removeEventListener("mousedown", handler);
  }, [dsDropOpen]);

  // ── Algorithm execution ──────────────────────────────────────────────────────
  const buildWithData = useCallback((data, features, targetCol) => {
    cancelRef.current   = true;
    growRef.current     = false;
    pausedAtRef.current = null;
    setGrowing(false);
    if (workerRef.current) { workerRef.current.terminate(); workerRef.current = null; }

    setRoundData([]);
    setCumulativeAccuracy([]);
    setCurRound(0);
    setCompletedCount(0);
    setRoundPhase(-1);
    setTreeStates({});
    setZoom(1);
    setBuildProgress(true);
    setSelectedSampleIdx(null);
    setWeightDisplayMode("input");

    const worker = new AdaBoostWorker();
    workerRef.current = worker;
    worker.onmessage = ({ data: { rounds, cumulativeAccuracy: ca } }) => {
      setBuildProgress(false);
      setRoundData(rounds);
      setCumulativeAccuracy(ca);
      workerRef.current = null;
      setRoundPhase(1); // start at "Building tree" phase — root node will appear via effect
      setHintDismissed(false);
      justRebuiltRef.current = true;
    };
    worker.onerror = () => { setBuildProgress(false); workerRef.current = null; };
    worker.postMessage({ data, features, targetCol, nRounds, maxDepth });
  }, [nRounds, maxDepth]);

  const buildForest = useCallback(() => {
    buildWithData(activeData, activeFeatures, activeTargetCol);
  }, [buildWithData, activeData, activeFeatures, activeTargetCol]);

  const switchToBuiltin = useCallback((key) => {
    if (growRef.current) { cancelRef.current = true; growRef.current = false; setGrowing(false); }
    setBuiltinDataset(key);
    setCustomDataset(null);
    setSelectedSampleIdx(null);
    const d = key === "music" ? musicData : heartData;
    const m = key === "music" ? musicMeta : heartMeta;
    buildWithData(d, m.features, m.targetCol);
  }, [buildWithData]);

  // Initial build
  useEffect(() => {
    const pending = location.state?.pendingCSV;
    if (pending) {
      const { data: rawRows, meta } = Papa.parse(pending.content, { header: true, skipEmptyLines: true });
      if (rawRows.length) {
        const headers = meta.fields;
        const naStats = detectNAs(rawRows, headers);
        setCsvModal({ fileName: pending.name, rawRows, headers, naStats,
          selectedTarget: headers[headers.length - 1],
          naStrategy: "drop", sampleMode: Math.min(1000, rawRows.length) });
        return;
      }
    }
    buildForest(); // eslint-disable-line react-hooks/exhaustive-deps
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  // ── Tree step system (same as RF) ─────────────────────────────────────────────
  const getTS = (idx) => treeStates[idx] || EMPTY_TS;
  const setTS = useCallback((idx, patch) => {
    setTreeStates(prev => { const old = prev[idx] || EMPTY_TS; return { ...prev, [idx]: { ...old, ...patch } }; });
  }, []);

  const getSteps = useCallback((rIdx) => {
    const tree = roundData[rIdx]?.tree;
    if (!tree) return [];
    const nodes = flattenNodes(tree).sort((a, b) => a.depth - b.depth);
    const steps = [];
    nodes.forEach(n => {
      if (n.type === "split") {
        steps.push({ nodeId: n.id, phase: 0 });
        steps.push({ nodeId: n.id, phase: 1 });
        steps.push({ nodeId: n.id, phase: 2, commit: true });
      } else {
        steps.push({ nodeId: n.id, phase: 0 });
        steps.push({ nodeId: n.id, phase: 1, commit: true });
      }
    });
    return steps;
  }, [roundData]);

  const goToStep = useCallback((rIdx, targetIdx) => {
    const steps = getSteps(rIdx);
    if (!steps.length) return;
    if (targetIdx < -1) targetIdx = -1;
    if (targetIdx >= steps.length) targetIdx = steps.length - 1;
    if (targetIdx === -1) { setTS(rIdx, EMPTY_TS); return; }
    const vis = [];
    for (let i = 0; i <= targetIdx; i++) {
      if (steps[i].commit && !vis.includes(steps[i].nodeId)) vis.push(steps[i].nodeId);
    }
    const s = steps[targetIdx];
    setTS(rIdx, { visibleIds: vis, nodeId: s.nodeId, phase: s.phase, stepIdx: targetIdx });
  }, [getSteps, setTS]);

  const instantComplete = useCallback((rIdx) => {
    const steps = getSteps(rIdx);
    if (!steps.length) return;
    const lastIdx = steps.length - 1;
    const vis = [];
    for (let i = 0; i <= lastIdx; i++) {
      if (steps[i].commit && !vis.includes(steps[i].nodeId)) vis.push(steps[i].nodeId);
    }
    const s = steps[lastIdx];
    setTS(rIdx, { visibleIds: vis, nodeId: s.nodeId, phase: s.phase, stepIdx: lastIdx });
  }, [getSteps, setTS]);

  // ── Phase advance/retreat ────────────────────────────────────────────────────
  const advanceStep = useCallback(() => {
    if (buildProgress || growing || roundPhase === 4) return;
    setHintDismissed(true);

    if (roundPhase === -1 || roundPhase === 0) {
      setRoundPhase(1);
      if (!treeStates[curRound]) goToStep(curRound, 0);
      return;
    }
    if (roundPhase === 1) {
      const steps = getSteps(curRound);
      const ts    = getTS(curRound);
      if (ts.stepIdx < steps.length - 1) { goToStep(curRound, ts.stepIdx + 1); }
      else                                { setRoundPhase(2); }
      return;
    }
    if (roundPhase === 2) { setRoundPhase(3); return; }
    if (roundPhase === 3) {
      setRoundPhase(4);
      setTimeout(() => {
        setRoundPhase(5);
        setCompletedCount(c => Math.max(c, curRound + 1));
      }, 600);
      return;
    }
    if (roundPhase === 5) {
      if (curRound < roundData.length - 1) {
        const next = curRound + 1;
        setCurRound(next);
        setRoundPhase(0);
        setWeightDisplayMode("input");
      }
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [buildProgress, growing, roundPhase, curRound, roundData, treeStates, getSteps, goToStep]);

  const retreatStep = useCallback(() => {
    if (buildProgress || growing || roundPhase === 4) return;
    if (roundPhase === 1) {
      const ts = getTS(curRound);
      if (ts.stepIdx > 0)       goToStep(curRound, ts.stepIdx - 1); // normal backward step
      else if (ts.stepIdx === 0) setTS(curRound, EMPTY_TS);          // remove root node, keep hint visible
      // stepIdx === -1: at hint-only state, nothing further to undo
      return;
    }
    if (roundPhase === 2) { setRoundPhase(1); return; }
    if (roundPhase === 3) { setRoundPhase(2); return; }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [buildProgress, growing, roundPhase, curRound, getTS, goToStep, setTS]);

  // After Rebuild: show root '?' node for round 0 once roundData is available
  useEffect(() => {
    if (justRebuiltRef.current && roundData.length > 0) {
      justRebuiltRef.current = false;
      goToStep(0, 0);
    }
  }, [roundData, goToStep]);

  // Weight display: transition from input → output during phase 4
  useEffect(() => {
    if (roundPhase !== 4) {
      setWeightDisplayMode(roundPhase >= 5 ? "output" : "input");
      return;
    }
    setWeightDisplayMode("input");
    const raf = requestAnimationFrame(() => requestAnimationFrame(() => setWeightDisplayMode("output")));
    return () => cancelAnimationFrame(raf);
  }, [roundPhase, curRound]);

  // ── Auto-grow ─────────────────────────────────────────────────────────────────
  const sleep = (ms) => new Promise(r => setTimeout(r, ms));

  const autoGrow = useCallback(async () => {
    if (growRef.current || !roundData.length) return;
    growRef.current   = true;
    cancelRef.current = false;
    setGrowing(true);

    const resume        = pausedAtRef.current;
    pausedAtRef.current = null;
    const startRoundIdx = resume?.roundIdx ?? completedCount;

    for (let rIdx = startRoundIdx; rIdx < roundData.length; rIdx++) {
      if (cancelRef.current) break;

      const isResumedRound = rIdx === startRoundIdx && resume !== null;
      const fromPhase      = isResumedRound ? resume.phase : 0;

      setCurRound(rIdx);

      // Phase 0: show input weights
      if (fromPhase <= 0) {
        liveStepRef.current = { roundIdx: rIdx, phase: 0, stepIdx: -1 };
        setRoundPhase(0);
        setWeightDisplayMode("input");
        await sleep(200 / speed);
        if (cancelRef.current) break;
      }

      // Phase 1: tree step animation
      if (fromPhase <= 1) {
        setRoundPhase(1);
        const steps    = getSteps(rIdx);
        const fromStep = (isResumedRound && resume.phase === 1) ? resume.stepIdx + 1 : 0;
        for (let i = fromStep; i < steps.length; i++) {
          if (cancelRef.current) break;
          liveStepRef.current = { roundIdx: rIdx, phase: 1, stepIdx: i };
          const vis = [];
          for (let j = 0; j <= i; j++) {
            if (steps[j].commit && !vis.includes(steps[j].nodeId)) vis.push(steps[j].nodeId);
          }
          setTS(rIdx, { visibleIds: vis, nodeId: steps[i].nodeId, phase: steps[i].phase, stepIdx: i });
          await sleep(180 / speed);
        }
      }
      if (cancelRef.current) break;

      // Phase 2: alpha display
      if (fromPhase <= 2) {
        liveStepRef.current = { roundIdx: rIdx, phase: 2, stepIdx: -1 };
        setRoundPhase(2);
        await sleep(450 / speed);
      }
      if (cancelRef.current) break;

      // Phase 3: error display
      if (fromPhase <= 3) {
        liveStepRef.current = { roundIdx: rIdx, phase: 3, stepIdx: -1 };
        setRoundPhase(3);
        await sleep(600 / speed);
      }

      // Phase 4: weight update animation
      if (fromPhase <= 4) {
        liveStepRef.current = { roundIdx: rIdx, phase: 4, stepIdx: -1 };
        setRoundPhase(4);
        setWeightDisplayMode("input");
        await sleep(50);
        setWeightDisplayMode("output");
        await sleep(650 / speed);
      }
      if (cancelRef.current) break;

      liveStepRef.current = { roundIdx: rIdx, phase: 5, stepIdx: -1 };
      setRoundPhase(5);
      setCompletedCount(rIdx + 1);
      await sleep(300 / speed);
    }

    growRef.current = false;
    setGrowing(false);
  }, [roundData, completedCount, speed, getSteps, setTS]);

  const growAllInstant = useCallback(() => {
    if (growing) return;
    roundData.forEach((_, rIdx) => instantComplete(rIdx));
    setCompletedCount(roundData.length);
    if (roundData.length > 0) {
      setCurRound(roundData.length - 1);
      setRoundPhase(5);
      setWeightDisplayMode("output");
    }
  }, [growing, roundData, instantComplete]);

  // ── Keyboard ──────────────────────────────────────────────────────────────────
  useEffect(() => {
    const h = (e) => {
      if (e.key === "ArrowRight" || e.key === "ArrowDown") { e.preventDefault(); advanceStep(); }
      else if (e.key === "ArrowLeft" || e.key === "ArrowUp") { e.preventDefault(); retreatStep(); }
    };
    window.addEventListener("keydown", h);
    return () => window.removeEventListener("keydown", h);
  });

  // ── Wheel zoom ────────────────────────────────────────────────────────────────
  useEffect(() => {
    const el = canvasRef.current;
    if (!el) return;
    const h = (e) => {
      e.preventDefault();
      const rect = el.getBoundingClientRect();
      const mx = e.clientX - rect.left, my = e.clientY - rect.top;
      let delta = -e.deltaY;
      if (e.ctrlKey) delta = delta * 0.008;
      else { const px = e.deltaMode === 1 ? delta * 16 : delta; delta = px * 0.001; }
      delta = Math.max(-0.15, Math.min(0.15, delta));
      const z = zoomLive.current, p = panLive.current;
      const nz = Math.max(0.08, Math.min(4, z * (1 + delta)));
      const sc = nz / z;
      setZoom(nz);
      setPan({ x: mx - sc * (mx - p.x), y: my - sc * (my - p.y) });
    };
    el.addEventListener("wheel", h, { passive: false });
    return () => el.removeEventListener("wheel", h);
  }, []);

  const onMouseDown = (e) => { isDragging.current = true; setCursorGrabbing(true); dragStart.current = { x: e.clientX, y: e.clientY, px: pan.x, py: pan.y }; };
  const onMouseMove = (e) => { if (!isDragging.current) return; setPan({ x: dragStart.current.px + e.clientX - dragStart.current.x, y: dragStart.current.py + e.clientY - dragStart.current.y }); };
  const onMouseUp   = () => { isDragging.current = false; setCursorGrabbing(false); };

  // ── Calculations sidebar resize/toggle (same as RF) ───────────────────────
  const onCalcHandleMouseDown = useCallback((e) => {
    if (e.button !== 0) return;
    e.preventDefault();
    const startX   = e.clientX;
    const startW   = calcPanelWidth;
    const expanded = calcExpanded;
    let moved = false;
    const onMove = (ev) => {
      if (!expanded) return;
      if (!moved && Math.abs(ev.clientX - startX) > 3) moved = true;
      if (!moved) return;
      const delta = startX - ev.clientX;
      setCalcPanelWidth(Math.max(180, Math.min(400, startW + delta)));
    };
    const onUp = () => {
      document.removeEventListener("mousemove", onMove);
      document.removeEventListener("mouseup", onUp);
      if (!moved) setCalcExpanded(x => !x);
    };
    document.addEventListener("mousemove", onMove);
    document.addEventListener("mouseup", onUp);
  }, [calcExpanded, calcPanelWidth]);

  // Center tree when round changes
  useEffect(() => {
    const tree = roundData[curRound]?.tree;
    const el   = canvasRef.current;
    if (!el || !tree) return;
    const canvasW = el.getBoundingClientRect().width;
    const tw = computeTreeWidth(tree);
    setPan({ x: (canvasW - tw) / 2, y: 20 });
  }, [curRound, roundData]);

  // File handling
  const openFile = useCallback((file) => {
    if (!file || !file.name.toLowerCase().endsWith(".csv")) return;
    const reader = new FileReader();
    reader.onload = (e) => {
      const { data: rawRows, meta } = Papa.parse(e.target.result, { header: true, skipEmptyLines: true });
      if (!rawRows.length) return;
      const headers = meta.fields;
      const naStats = detectNAs(rawRows, headers);
      setCsvModal({ fileName: file.name, rawRows, headers, naStats,
        selectedTarget: headers[headers.length - 1],
        naStrategy: "drop", sampleMode: Math.min(1000, rawRows.length) });
    };
    reader.readAsText(file);
  }, []);

  const handleDrop = useCallback((e) => {
    e.preventDefault(); setDragOver(false); openFile(e.dataTransfer.files[0]);
  }, [openFile]);

  const handleDataConfirm = useCallback((data, features, targetCol, name, totalRows, sampledRows, classLabels, taskType = "classification") => {
    if (taskType === "regression") {
      setCsvError({ type: "regression", name });
      setCsvModal(null);
      return;
    }
    const uniqueClasses = [...new Set(data.map(r => r[targetCol]))];
    if (uniqueClasses.length > 2) {
      setCsvError({ type: "multiclass", name, nClasses: uniqueClasses.length });
      setCsvModal(null);
      return;
    }
    setCsvError(null);
    setCustomDataset({ data, features, targetCol, name, totalRows, sampledRows, classLabels });
    setSelectedSampleIdx(null);
    setCsvModal(null);
    buildWithData(data, features, targetCol);
  }, [buildWithData]);

  // Scroll hint after all rounds complete
  useEffect(() => {
    if (completedCount < roundData.length || !roundData.length) { setScrollHint(false); return; }
    setScrollHint(true);
    const timer = setTimeout(() => setScrollHint(false), 5000);
    const el = predSectionRef.current;
    if (!el) return () => clearTimeout(timer);
    const obs = new IntersectionObserver(([entry]) => {
      if (entry.isIntersecting) { setScrollHint(false); clearTimeout(timer); }
    }, { threshold: 0.1 });
    obs.observe(el);
    return () => { clearTimeout(timer); obs.disconnect(); };
  }, [completedCount, roundData.length]);

  // ── Derived values ─────────────────────────────────────────────────────────────
  const currentRoundData = roundData[curRound];
  const currentTree = currentRoundData?.tree ?? null;
  const treeWidth   = currentTree ? computeTreeWidth(currentTree) : X_PAD * 2 + 2 * LEAF_SPACING;
  const svgH        = (maxDepth + 1) * Y_GAP + 60;
  const allNodes    = currentTree ? flattenNodes(currentTree) : [];
  const positions   = currentTree ? computePositions(currentTree) : {};
  const ts          = getTS(curRound);
  const visibleSet  = new Set(ts.visibleIds);
  const currentNode = allNodes.find(n => n.id === ts.nodeId);
  const totalSteps  = getSteps(curRound).length;

  // Current feature pool data (from the node being animated)
  const fpNode = allNodes.find(n => n.id === ts.nodeId && n.type === "split") ?? null;
  const fpEvals = fpNode?.allFeatureEvals ?? [];
  const fpCandidates = new Set(fpNode?.candidateIndices ?? []);
  const fpChosen    = roundPhase === 1 && ts.phase >= 2 ? fpNode?.featureIndex ?? -1 : -1;

  // Sample prediction
  const safeSampleIdx   = selectedSampleIdx !== null && selectedSampleIdx < activeData.length ? selectedSampleIdx : null;
  const selectedSample  = safeSampleIdx !== null ? activeData[safeSampleIdx] : null;

  const samplePath = useMemo(() => {
    if (!selectedSample || !currentTree) return new Set();
    return new Set(getSamplePath(currentTree, selectedSample, activeFeatures));
  }, [selectedSample, currentTree, activeFeatures]);

  // Per-round alpha predictions for selected sample
  const sampleRoundPreds = useMemo(() => {
    if (!selectedSample || !roundData.length) return [];
    return roundData.slice(0, completedCount).map((rd, i) => ({
      round: i + 1,
      alpha: rd.alpha,
      pred: predictAdaTree(rd.tree, selectedSample, activeFeatures),
    }));
  }, [selectedSample, roundData, completedCount, activeFeatures]);

  const ensembleScores = useMemo(() => {
    if (!selectedSample || !completedCount) return null;
    return adaBoostEnsembleScores(
      roundData.slice(0, completedCount),
      selectedSample,
      activeFeatures
    );
  }, [selectedSample, roundData, completedCount, activeFeatures]);

  const ensemblePrediction = ensembleScores
    ? Object.entries(ensembleScores).sort((a, b) => b[1] - a[1])[0]?.[0]
    : null;
  const totalAlpha = ensembleScores ? Object.values(ensembleScores).reduce((s, v) => s + v, 0) : 0;

  // Weight display
  const displayWeights = useMemo(() => {
    if (!currentRoundData) return null;
    return weightDisplayMode === "output"
      ? currentRoundData.weightsOutput
      : currentRoundData.weightsInput;
  }, [currentRoundData, weightDisplayMode]);

  const sampleClassColors = useMemo(() => {
    if (!displayWeights) return [];
    return activeData.map(row => classColor(row[activeTargetCol], allClasses));
  }, [activeData, activeTargetCol, allClasses, displayWeights]);

  // For phase 2+: color bars green/red based on current round's predictions
  const predCorrect = useMemo(() => {
    if (!currentRoundData || roundPhase < 2) return null;
    return currentRoundData.misclassified.map(m => !m);
  }, [currentRoundData, roundPhase]);

  const inp = {
    padding: "6px 10px", borderRadius: 8, background: "#161c2a",
    border: "none", color: C.text, fontSize: 12, fontFamily: "'JetBrains Mono',monospace",
    outline: "none", boxShadow: "inset 0 0 0 1px rgba(255,255,255,0.06)",
  };

  const disabled = buildProgress || growing;

  // ── Render ────────────────────────────────────────────────────────────────────
  return (
    <div
      style={{ minHeight: "100vh", background: C.bg, color: C.text }}
      onDragOver={e => { e.preventDefault(); setDragOver(true); }}
      onDragLeave={() => setDragOver(false)}
      onDrop={handleDrop}
    >
      {/* Build overlay */}
      {buildProgress && (
        <div style={{
          position: "fixed", inset: 0, zIndex: 200, background: "rgba(10,14,23,0.82)",
          backdropFilter: "blur(5px)", display: "flex", flexDirection: "column",
          alignItems: "center", justifyContent: "center", gap: 18, pointerEvents: "none",
        }}>
          <div style={{ fontSize: 15, fontWeight: 700, color: C.text }}>Training AdaBoost…</div>
          <div style={{ width: 280, height: 4, background: C.border, borderRadius: 2, overflow: "hidden" }}>
            <div style={{ height: "100%", width: "60%", background: `linear-gradient(90deg,${C.accent},${C.green})`,
              borderRadius: 2, animation: "abFadeIn 1s ease infinite alternate" }} />
          </div>
          <div style={{ fontSize: 10, color: C.dim }}>{nRounds} rounds · max depth {maxDepth}</div>
        </div>
      )}

      {/* Dataset dropdown menu — portal to escape GlobalHeader stacking context */}
      {dsDropOpen && dsDropPos && createPortal(
        <div ref={dsMenuRef} className="ab-ds-drop-menu"
          style={{ top: dsDropPos.top, left: dsDropPos.left }}>
          {/* Heart Disease — active */}
          <button className="ab-ds-opt"
            data-active={!customDataset && builtinDataset === "heart" ? "true" : "false"}
            onClick={() => { setDsDropOpen(false); switchToBuiltin("heart"); }}>
            {!customDataset && builtinDataset === "heart" && <span style={{ fontSize: 9 }}>✓ </span>}
            Built-in: Heart Disease (Binary classification)
          </button>

          {/* Music Genres — disabled: multiclass */}
          <button className="ab-ds-opt" data-disabled="true"
            onMouseEnter={e => {
              const r = e.currentTarget.getBoundingClientRect();
              setDsOptTooltip({ ...tooltipPosition(r, { prefer: "above", width: 280, height: 58 }),
                text: "Our AdaBoost implementation currently supports binary classification only. For multiclass, try Decision Tree, Bagging, or Random Forest." });
            }}
            onMouseLeave={() => setDsOptTooltip(null)}>
            🔒 Built-in: Music Genres (Multiclass — not supported)
          </button>

          {/* Salary — disabled: regression */}
          <button className="ab-ds-opt" data-disabled="true"
            onMouseEnter={e => {
              const r = e.currentTarget.getBoundingClientRect();
              setDsOptTooltip({ ...tooltipPosition(r, { prefer: "above", width: 280, height: 58 }),
                text: "Our AdaBoost implementation currently supports classification only. For regression, try Decision Tree, Bagging, or Random Forest." });
            }}
            onMouseLeave={() => setDsOptTooltip(null)}>
            🔒 Built-in: Salary (Regression — not supported)
          </button>

          {/* Custom dataset if loaded */}
          {customDataset && (
            <button className="ab-ds-opt" data-active="true"
              onClick={() => setDsDropOpen(false)}>
              ✓ {customDataset.name}
            </button>
          )}

          {/* Upload CSV */}
          <div style={{ borderTop: "1px solid rgba(255,255,255,0.06)", margin: "4px 0" }} />
          <button className="ab-ds-opt"
            onClick={() => { setDsDropOpen(false); fileInputRef.current?.click(); }}>
            ↑ Upload CSV…
          </button>
        </div>,
        document.body
      )}

      {/* Dataset option tooltip — portal to escape stacking context */}
      {dsOptTooltip && createPortal(
        <div style={{
          position: "fixed", top: dsOptTooltip.top, left: dsOptTooltip.left,
          zIndex: 99999, maxWidth: 280, padding: "8px 12px",
          background: "#1a2235", border: "1px solid rgba(255,255,255,0.12)",
          borderRadius: 8, fontSize: 11, color: C.text, lineHeight: 1.55,
          boxShadow: "0 4px 20px rgba(0,0,0,0.6)", pointerEvents: "none",
        }}>
          {dsOptTooltip.text}
        </div>,
        document.body
      )}

      {/* Drag overlay */}
      {dragOver && (
        <div style={{ position: "fixed", inset: 0, zIndex: 50, background: `${C.accent}18`,
          border: `2px dashed ${C.accent}`, borderRadius: 8, pointerEvents: "none",
          display: "flex", alignItems: "center", justifyContent: "center" }}>
          <span style={{ fontSize: 18, color: C.accent, fontWeight: 700 }}>Drop CSV to load dataset</span>
        </div>
      )}

      {csvModal && (
        <DataModal modal={csvModal}
          onUpdate={patch => setCsvModal(prev => ({ ...prev, ...patch }))}
          onConfirm={handleDataConfirm}
          onCancel={() => setCsvModal(null)} />
      )}

      <input ref={fileInputRef} type="file" accept=".csv" style={{ display: "none" }}
        onChange={e => { openFile(e.target.files[0]); e.target.value = ""; }} />

      {/* ── Header ─────────────────────────────────────────────────────────── */}
      <GlobalHeader
        right={
          <button onClick={() => fileInputRef.current?.click()}
            style={{ background: "none", border: "none", cursor: "pointer", fontSize: 12,
              color: C.dim, fontWeight: 500, padding: 0, transition: "color 0.15s", fontFamily: "inherit" }}
            onMouseEnter={e => { e.currentTarget.style.color = C.text; }}
            onMouseLeave={e => { e.currentTarget.style.color = C.dim; }}>
            ↑ Upload CSV
          </button>
        }
        infoBar={(() => {
          const targetDisp = customDataset?.originalTarget ?? "Disease";
          const nClasses   = allClasses.length;
          const taskTag    = nClasses === 2 ? "Binary" : `${nClasses}-class`;
          const Badge = ({ children }) => (
            <span style={{ display: "inline-flex", alignItems: "center", background: "rgba(255,255,255,0.07)",
              borderRadius: 6, padding: "4px 10px", fontSize: 11, whiteSpace: "nowrap" }}>{children}</span>
          );
          return (
            <>
              <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                <span style={{ fontSize: 18, fontWeight: 700, color: C.text, letterSpacing: "-0.3px" }}>AdaBoost</span>
                <span style={{ fontSize: 10, color: C.dim }}>boosting ensemble · sequential rounds · alpha-weighted voting</span>
              </div>
              <div style={{ display: "flex", alignItems: "center", gap: 8, flexShrink: 0 }}>
                <div style={{ display: "flex", alignItems: "center", gap: 7 }}>
                  <span style={{ fontSize: 9, color: C.muted, letterSpacing: "0.04em" }}>Dataset</span>
                  <div ref={dsDropRef} style={{ position: "relative", display: "inline-flex", alignItems: "center" }}>
                    {/* Trigger button */}
                    <button
                      disabled={disabled}
                      onClick={e => {
                        if (disabled) return;
                        const r = e.currentTarget.getBoundingClientRect();
                        setDsDropPos({ top: r.bottom + 4, left: r.left });
                        setDsDropOpen(v => !v);
                      }}
                      style={{
                        WebkitAppearance: "none", appearance: "none",
                        background: "rgba(255,255,255,0.05)", border: "1px solid rgba(255,255,255,0.1)",
                        borderRadius: 20, color: C.text, fontFamily: "inherit", fontSize: 12,
                        fontWeight: 500, padding: "5px 28px 5px 12px", cursor: disabled ? "not-allowed" : "pointer",
                        outline: "none", transition: "box-shadow 0.2s ease, border-color 0.2s ease",
                        opacity: disabled ? 0.5 : 1,
                      }}
                      onMouseEnter={e => { if (!disabled) { e.currentTarget.style.borderColor = "rgba(255,255,255,0.2)"; e.currentTarget.style.boxShadow = "0 0 0 3px rgba(245,158,11,0.12)"; }}}
                      onMouseLeave={e => { e.currentTarget.style.borderColor = "rgba(255,255,255,0.1)"; e.currentTarget.style.boxShadow = "none"; }}
                    >
                      {customDataset
                        ? customDataset.name
                        : "Built-in: Heart Disease (Binary classification)"}
                    </button>
                    <span style={{ position: "absolute", right: 9, top: "50%", transform: "translateY(-50%)",
                      pointerEvents: "none", color: C.dimmer, fontSize: 10 }}>▾</span>
                  </div>
                </div>
                <Badge><span style={{ color: C.dim, marginRight: 4 }}>Target</span><span style={{ color: C.text, fontWeight: 600 }}>{targetDisp}</span></Badge>
                <Badge><span style={{ color: C.text, fontWeight: 600 }}>{taskTag}</span></Badge>
                <Badge><span style={{ color: C.text, fontWeight: 600 }}>{activeData.length.toLocaleString()}</span><span style={{ color: C.dim, marginLeft: 4 }}>samples</span></Badge>
              </div>
            </>
          );
        })()}
      />

      {/* ── CSV incompatibility error banner ───────────────────────────────── */}
      {csvError && (
        <div style={{
          padding: "11px 20px", background: `${C.red}12`,
          borderBottom: `1px solid ${C.red}33`,
          display: "flex", alignItems: "flex-start", gap: 12,
        }}>
          <span style={{ fontSize: 15, lineHeight: "18px", flexShrink: 0 }}>⚠</span>
          <div style={{ flex: 1, fontSize: 11, color: C.text, lineHeight: 1.6 }}>
            {csvError.type === "regression" ? (
              <>
                <strong style={{ color: C.orange }}>{csvError.name}</strong> appears to be a{" "}
                <strong>regression task</strong>. Our AdaBoost implementation currently supports{" "}
                <strong>classification only</strong>. For regression, try{" "}
                <span style={{ color: C.accent }}>Decision Tree</span>,{" "}
                <span style={{ color: C.accent }}>Bagging</span>, or{" "}
                <span style={{ color: C.accent }}>Random Forest</span>.
              </>
            ) : (
              <>
                <strong style={{ color: C.orange }}>{csvError.name}</strong> has{" "}
                <strong>{csvError.nClasses} classes</strong>. Our AdaBoost implementation currently supports{" "}
                <strong>binary classification only</strong>. For multiclass classification, try{" "}
                <span style={{ color: C.accent }}>Decision Tree</span>,{" "}
                <span style={{ color: C.accent }}>Bagging</span>, or{" "}
                <span style={{ color: C.accent }}>Random Forest</span>.
              </>
            )}
          </div>
          <button onClick={() => setCsvError(null)} style={{
            background: "none", border: "none", color: C.dim, cursor: "pointer",
            fontSize: 14, lineHeight: 1, padding: "2px 4px", flexShrink: 0,
            transition: "color 0.15s", fontFamily: "inherit",
          }}
            onMouseEnter={e => { e.currentTarget.style.color = C.text; }}
            onMouseLeave={e => { e.currentTarget.style.color = C.dim; }}>
            ✕
          </button>
        </div>
      )}

      {/* ── Controls toolbar ────────────────────────────────────────────────── */}
      <div style={{ display: "flex", alignItems: "flex-end", gap: 10, padding: "12px 20px",
        background: C.panel, boxShadow: "0 1px 0 rgba(255,255,255,0.03), 0 4px 20px rgba(0,0,0,0.35)",
        flexWrap: "wrap", position: "relative", zIndex: 20 }}>

        <div style={{ display: "flex", alignItems: "flex-end", gap: 12, flex: 1, flexWrap: "wrap" }}>
          {/* Max depth */}
          <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
            <label style={{ fontSize: 9, color: C.muted, fontWeight: 400 }}>Max depth</label>
            <input type="number" min={1} max={3} value={maxDepthStr} disabled={disabled}
              onChange={e => setMaxDepthStr(e.target.value)}
              onKeyDown={e => e.stopPropagation()}
              onBlur={() => { const v = Math.max(1, Math.min(3, parseInt(maxDepthStr, 10) || 1)); setMaxDepth(v); setMaxDepthStr(String(v)); }}
              style={{ ...inp, width: 58 }} />
          </div>

          {/* Max features — locked */}
          <div style={{ position: "relative", cursor: "default" }}
            onMouseEnter={e => setLockedTip(tooltipPosition(e.currentTarget.getBoundingClientRect(), { prefer: 'above', width: 240, height: 52 }))}
            onMouseLeave={() => setLockedTip(null)}>
            <div style={{ display: "flex", flexDirection: "column", gap: 4, opacity: 0.45, pointerEvents: "none", userSelect: "none" }}>
              <label style={{ fontSize: 9, color: C.muted, fontWeight: 400, display: "flex", alignItems: "center", gap: 4 }}>
                Max features <span style={{ fontSize: 9 }}>🔒</span>
              </label>
              <select disabled style={{ ...inp, cursor: "not-allowed", width: "auto" }}>
                <option>p (all) → {activeFeatures.length}</option>
              </select>
            </div>
          </div>

          {/* Rounds */}
          <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
            <label style={{ fontSize: 9, color: C.muted, fontWeight: 400 }}>Rounds</label>
            <input type="number" min={1} max={50} value={nRoundsStr} disabled={disabled}
              onChange={e => setNRoundsStr(e.target.value)}
              onKeyDown={e => e.stopPropagation()}
              onBlur={() => { const v = Math.max(1, Math.min(50, parseInt(nRoundsStr, 10) || 1)); setNRounds(v); setNRoundsStr(String(v)); }}
              style={{ ...inp, width: 58 }} />
          </div>

          {/* Speed */}
          <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
            <label style={{ fontSize: 9, color: C.muted, fontWeight: 400 }}>Speed</label>
            <select value={speed} onChange={e => setSpeed(Number(e.target.value))} style={{ ...inp, cursor: "pointer" }}>
              <option value={0.5}>0.5×</option>
              <option value={1}>1×</option>
              <option value={2}>2×</option>
              <option value={4}>4×</option>
            </select>
          </div>
        </div>

        {/* Buttons */}
        <div style={{ display: "flex", gap: 8, alignItems: "flex-end" }}>
          <button disabled={disabled} onClick={() => buildForest()}
            style={{ padding: "7px 14px", borderRadius: 8, border: `1px solid ${C.border}`,
              background: "transparent", color: C.dim, fontSize: 11, fontFamily: "inherit",
              cursor: disabled ? "default" : "pointer", opacity: disabled ? 0.4 : 1,
              transition: "color 0.15s, border-color 0.15s" }}
            onMouseEnter={e => { if (!disabled) { e.currentTarget.style.color = C.text; e.currentTarget.style.borderColor = C.dim; } }}
            onMouseLeave={e => { e.currentTarget.style.color = C.dim; e.currentTarget.style.borderColor = C.border; }}>
            Rebuild
          </button>

          {growing ? (
            <button onClick={() => { pausedAtRef.current = { ...liveStepRef.current }; cancelRef.current = true; growRef.current = false; setGrowing(false); }}
              style={{ padding: "7px 14px", borderRadius: 8, border: "none",
                background: `${C.red}22`, color: C.red, fontSize: 11, fontFamily: "inherit", cursor: "pointer",
                transition: "background 0.15s, box-shadow 0.15s" }}
              onMouseEnter={e => { e.currentTarget.style.background = `${C.red}38`; e.currentTarget.style.boxShadow = "0 0 10px rgba(239,68,68,0.2)"; }}
              onMouseLeave={e => { e.currentTarget.style.background = `${C.red}22`; e.currentTarget.style.boxShadow = "none"; }}>
              Stop
            </button>
          ) : (
            <button disabled={!roundData.length || buildProgress} onClick={() => {
                setHintDismissed(true);
                // If already in tree-building phase, skip the weights animation
                if (roundPhase === 1) {
                  pausedAtRef.current = { roundIdx: curRound, phase: 1, stepIdx: ts.stepIdx };
                }
                autoGrow();
              }}
              style={{ padding: "7px 18px", borderRadius: 8, border: "none",
                background: !roundData.length || buildProgress ? C.border : `linear-gradient(135deg,${C.accent},#d97706)`,
                color: !roundData.length || buildProgress ? C.dim : "#000",
                fontSize: 11, fontFamily: "inherit", cursor: !roundData.length || buildProgress ? "default" : "pointer",
                fontWeight: 600, transition: "box-shadow 0.15s, filter 0.15s" }}
              onMouseEnter={e => { if (!(!roundData.length || buildProgress)) { e.currentTarget.style.boxShadow = "0 0 12px rgba(245,158,11,0.3)"; e.currentTarget.style.filter = "brightness(1.1)"; } }}
              onMouseLeave={e => { e.currentTarget.style.boxShadow = "none"; e.currentTarget.style.filter = "none"; }}>
              Grow
            </button>
          )}

          <button disabled={!roundData.length || growing || buildProgress} onClick={growAllInstant}
            style={{ padding: "7px 14px", borderRadius: 8, border: `1px solid ${C.border}`,
              background: "transparent", color: C.dim, fontSize: 11, fontFamily: "inherit",
              cursor: !roundData.length || growing || buildProgress ? "default" : "pointer",
              opacity: !roundData.length || growing || buildProgress ? 0.4 : 1,
              transition: "color 0.15s, border-color 0.15s" }}
            onMouseEnter={e => { if (!(!roundData.length || growing || buildProgress)) { e.currentTarget.style.color = C.text; e.currentTarget.style.borderColor = C.dim; } }}
            onMouseLeave={e => { e.currentTarget.style.color = C.dim; e.currentTarget.style.borderColor = C.border; }}>
            Complete All
          </button>
        </div>
      </div>

      {/* ── Step 1: Training ────────────────────────────────────────────────── */}
      <div style={{ padding: "12px 20px 4px 20px" }}>
        <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
          <span style={{ fontSize: 11, fontWeight: 700, color: C.accent }}>Step 1</span>
          <span style={{ fontSize: 11, fontWeight: 700, color: C.text }}>Training</span>
          <span style={{ fontSize: 9, color: C.dim, marginLeft: 4 }}>←→ arrow keys to step · Double-click round tab to complete</span>
        </div>
      </div>

      {/* ── Round tabs ──────────────────────────────────────────────────────── */}
      {roundData.length > 0 && (
        <div style={{ padding: "6px 20px 0", display: "flex", gap: 3, overflowX: "auto",
          scrollbarWidth: "none", flexWrap: "nowrap" }}>
          {roundData.map((rd, rIdx) => {
            const isComplete  = rIdx < completedCount;
            const isCurrent   = rIdx === curRound;
            const isLocked    = rIdx > completedCount;
            const isNext      = rIdx === completedCount && !isComplete;
            const tipText     = isComplete
              ? `R${rIdx + 1}  ·  α = ${rd.alpha}  ·  ε = ${(rd.error * 100).toFixed(1)}%`
              : isLocked ? `R${rIdx + 1} — locked (complete previous rounds first)` : `R${rIdx + 1}`;
            return (
              <button
                key={rIdx}
                title={tipText}
                disabled={isLocked}
                onDoubleClick={() => {
                  if (isLocked) return;
                  instantComplete(rIdx);
                  setCompletedCount(c => Math.max(c, rIdx + 1));
                  setCurRound(rIdx);
                  setRoundPhase(5);
                  setWeightDisplayMode("output");
                }}
                onClick={() => {
                  if (isLocked) return;
                  setCurRound(rIdx);
                  if (isComplete) { setRoundPhase(5); setWeightDisplayMode("output"); }
                }}
                style={{
                  padding: "3px 8px", borderRadius: 20, border: "none",
                  cursor: isLocked ? "not-allowed" : "pointer",
                  fontSize: 10, fontWeight: isCurrent ? 700 : 400,
                  fontFamily: "'JetBrains Mono',monospace",
                  background: isCurrent
                    ? (isComplete ? `${C.green}28` : `${C.accent}28`)
                    : "rgba(255,255,255,0.04)",
                  color: isLocked ? C.dimmer
                    : isCurrent ? (isComplete ? C.green : C.accent)
                    : isComplete ? C.green : isNext ? C.text : C.dimmer,
                  boxShadow: isCurrent
                    ? `inset 0 0 0 1px ${isComplete ? C.green : C.accent}55`
                    : "inset 0 0 0 1px rgba(255,255,255,0.06)",
                  opacity: isLocked ? 0.3 : 1,
                  transition: "background 0.15s, color 0.15s, box-shadow 0.15s",
                  whiteSpace: "nowrap", flexShrink: 0, lineHeight: 1.4,
                }}
                onMouseEnter={e => { if (!isLocked) e.currentTarget.style.background = isCurrent ? (isComplete ? `${C.green}40` : `${C.accent}40`) : "rgba(255,255,255,0.09)"; }}
                onMouseLeave={e => { e.currentTarget.style.background = isCurrent ? (isComplete ? `${C.green}28` : `${C.accent}28`) : "rgba(255,255,255,0.04)"; }}
              >
                {isComplete ? `✓ R${rIdx + 1}` : `R${rIdx + 1}`}
              </button>
            );
          })}
        </div>
      )}

      {/* ── Phase indicator ─────────────────────────────────────────────────── */}
      {roundPhase >= 0 && roundData.length > 0 && (
        <div style={{ padding: "6px 20px", display: "flex", gap: 6, alignItems: "center" }}>
          {PHASE_LABELS.map((p, i) => {
            const active = roundPhase === i;
            const done   = (i < roundPhase) || (roundPhase === 5 && i < 5);
            return (
              <div key={i} style={{ display: "flex", alignItems: "center", gap: 4 }}>
                <div style={{
                  padding: "2px 8px", borderRadius: 20, fontSize: 9, fontWeight: active ? 700 : 400,
                  background: active ? `${p.color}22` : "transparent",
                  color: active ? p.color : done ? `${p.color}77` : C.dimmer,
                  boxShadow: active ? `inset 0 0 0 1px ${p.color}44` : "none",
                  transition: "all 0.2s",
                }}>
                  {p.icon} {p.text}
                </div>
                {i < 5 && <span style={{ color: C.dimmer, fontSize: 8 }}>→</span>}
              </div>
            );
          })}
          {/* Reset — single click: current round · double click: all rounds */}
          <button
            disabled={growing}
            onMouseEnter={e => setResetTooltip(tooltipPosition(e.currentTarget.getBoundingClientRect(), { prefer: 'above', width: 278, height: 28 }))}
            onMouseLeave={() => setResetTooltip(null)}
            onClick={() => {
              if (resetClickRef.current) return;
              resetClickRef.current = setTimeout(() => {
                resetClickRef.current = null;
                setTS(curRound, EMPTY_TS);
                setRoundPhase(0);
              }, 220);
            }}
            onDoubleClick={() => {
              clearTimeout(resetClickRef.current);
              resetClickRef.current = null;
              pausedAtRef.current = null;
              setTreeStates({});
              setCurRound(0);
              setCompletedCount(0);
              setRoundPhase(0);
              setWeightDisplayMode("input");
            }}
            style={{
              marginLeft: "auto", padding: "2px 10px", borderRadius: 20, fontSize: 9,
              background: "rgba(255,255,255,0.04)", border: "1px solid rgba(255,255,255,0.08)",
              color: growing ? C.dimmer : C.dim, cursor: growing ? "default" : "pointer",
              fontFamily: "inherit", opacity: growing ? 0.4 : 1,
              transition: "background 0.15s, border-color 0.15s, color 0.15s",
            }}
            onMouseEnter={e => { if (!growing) { e.currentTarget.style.background = "rgba(255,255,255,0.09)"; e.currentTarget.style.borderColor = "rgba(255,255,255,0.18)"; e.currentTarget.style.color = C.text; } }}
            onMouseLeave={e => { e.currentTarget.style.background = "rgba(255,255,255,0.04)"; e.currentTarget.style.borderColor = "rgba(255,255,255,0.08)"; e.currentTarget.style.color = growing ? C.dimmer : C.dim; }}>
            Reset
          </button>
        </div>
      )}

      {/* ── Main canvas row: left sidebar + tree canvas + calculations ─────── */}
      <div style={{ display: "flex", height: "calc(100vh - 370px)", minHeight: 380, overflow: "hidden" }}>

        {/* Left sidebar: Training Error chart + Sample Weights chart */}
        <div style={{ width: 240, flexShrink: 0, borderRight: `1px solid ${C.border}`,
          display: "flex", flexDirection: "column", overflowY: "auto", overflowX: "hidden",
          scrollbarWidth: "thin", scrollbarColor: `${C.dimmer} transparent` }}>

          {/* Training Error chart */}
          <div style={{ flexShrink: 0, padding: "10px 12px", borderBottom: `1px solid ${C.border}` }}>
            <div style={{ fontSize: 9, fontWeight: 600, color: C.dim, textTransform: "uppercase",
              letterSpacing: "0.06em", marginBottom: 6 }}>Training Error</div>
            <ErrorChart accuracies={cumulativeAccuracy} completedCount={completedCount} />
          </div>

          {/* Sample Weights chart */}
          <div style={{ flexShrink: 0, padding: "10px 12px" }}>
            <div style={{ fontSize: 9, fontWeight: 600, color: C.dim, textTransform: "uppercase",
              letterSpacing: "0.06em", marginBottom: 6 }}>
              Sample Weights
              {currentRoundData && <span style={{ fontWeight: 400, marginLeft: 6, textTransform: "none" }}>
                R{curRound + 1}
              </span>}
            </div>
            {displayWeights ? (
              <>
                <WeightChart
                  weights={displayWeights}
                  sortWeights={currentRoundData?.weightsInput}
                  classColors={sampleClassColors}
                  predCorrect={predCorrect}
                  showPredColors={roundPhase >= 2}
                  maxH={120}
                />
                <div style={{ marginTop: 6, display: "flex", gap: 10, flexWrap: "wrap" }}>
                  {roundPhase >= 2 ? (
                    <>
                      <div style={{ display: "flex", alignItems: "center", gap: 4, fontSize: 8, color: C.dim }}>
                        <div style={{ width: 8, height: 8, background: "#10b981", borderRadius: 2 }} /> Correct
                      </div>
                      <div style={{ display: "flex", alignItems: "center", gap: 4, fontSize: 8, color: C.dim }}>
                        <div style={{ width: 8, height: 8, background: "#ef4444", borderRadius: 2 }} /> Misclassified
                      </div>
                    </>
                  ) : allClasses.slice(0, 3).map((cls, i) => (
                    <div key={cls} style={{ display: "flex", alignItems: "center", gap: 4, fontSize: 8, color: C.dim }}>
                      <div style={{ width: 8, height: 8, background: CLASS_PALETTE[i], borderRadius: 2 }} />
                      {cls}
                    </div>
                  ))}
                </div>
                {currentRoundData && roundPhase >= 2 && (
                  <div style={{ marginTop: 6, fontSize: 8, color: C.dim }}>
                    <span style={{ color: C.green }}>{currentRoundData.misclassified.filter(m => !m).length} correct</span>
                    {" · "}
                    <span style={{ color: C.red }}>{currentRoundData.misclassified.filter(m => m).length} misclassified</span>
                  </div>
                )}
                {roundPhase >= 5 && currentRoundData && (() => {
                  const avgW = 1 / activeData.length;
                  const aboveAvg = currentRoundData.weightsOutput.filter(w => w > avgW).length;
                  return <div style={{ marginTop: 4, fontSize: 8, color: C.dim }}>{aboveAvg} samples above avg weight</div>;
                })()}
              </>
            ) : (
              <div style={{ fontSize: 9, color: C.dimmer, marginTop: 8 }}>Build to see weights</div>
            )}
          </div>
        </div>

        {/* Center: Tree canvas */}
        <div style={{ flex: 1, overflow: "hidden", position: "relative" }}>
          {/* Ghost tree overlay: show previous round's tree when in Weights phase (round 2+) */}
          {roundPhase === 0 && curRound > 0 && (() => {
            const prevRd = roundData[curRound - 1];
            if (!prevRd?.tree) return null;
            const prevTree = prevRd.tree;
            const prevPos  = computePositions(prevTree);
            const prevNodes = flattenNodes(prevTree);
            const prevW    = computeTreeWidth(prevTree);
            const prevH    = (maxDepth + 1) * Y_GAP + 60;
            const canvasEl = canvasRef.current;
            const canvasW  = canvasEl ? canvasEl.getBoundingClientRect().width : 600;
            const ghostPanX = (canvasW - prevW) / 2;
            return (
              <div style={{ position: "absolute", inset: 0, pointerEvents: "none", zIndex: 0, overflow: "hidden" }}>
                <svg width={prevW} height={prevH}
                  style={{ transform: `translate(${ghostPanX}px,20px)`, transformOrigin: "0 0",
                    overflow: "visible", opacity: 0.18 }}>
                  {prevNodes.filter(n => n.type === "split").flatMap(n => [
                    { parent: n, child: n.left }, { parent: n, child: n.right },
                  ]).map(({ parent, child }) => {
                    const p1 = prevPos[parent.id], p2 = prevPos[child.id];
                    if (!p1 || !p2) return null;
                    return <line key={`${parent.id}-${child.id}`} x1={p1.x} y1={p1.y + 27} x2={p2.x} y2={p2.y - 24}
                      stroke={C.edge} strokeWidth={1.4} strokeDasharray="4 2" />;
                  })}
                  {prevNodes.map(n => {
                    const pos = prevPos[n.id];
                    if (!pos) return null;
                    const { x, y } = pos;
                    if (n.type === "leaf") {
                      const predColor = classColor(n.prediction, allClasses);
                      return (
                        <g key={n.id}>
                          <rect x={x - 38} y={y - 24} width={76} height={50} rx={11}
                            fill={C.panel} stroke={`${predColor}88`} strokeWidth={1.4} />
                          <text x={x} y={y - 9} textAnchor="middle" fill={C.text}
                            fontSize={9} fontFamily={MONO} fontWeight={600}>{String(n.prediction)}</text>
                        </g>
                      );
                    }
                    return (
                      <g key={n.id}>
                        <rect x={x - 90} y={y - 27} width={180} height={54} rx={11}
                          fill={C.panel} stroke={C.accent} strokeWidth={1.4} />
                        <text x={x} y={y - 6} textAnchor="middle" fill={C.accent}
                          fontSize={8.5} fontFamily={MONO} fontWeight={700}>{n.featureName ?? ""}</text>
                        <text x={x} y={y + 10} textAnchor="middle" fill={C.dim}
                          fontSize={7.5} fontFamily={MONO}>≤{n.threshold} n={n.samples}</text>
                      </g>
                    );
                  })}
                </svg>
                <div style={{ position: "absolute", bottom: 8, left: "50%", transform: "translateX(-50%)",
                  fontSize: 9, color: C.dimmer, fontFamily: MONO, whiteSpace: "nowrap" }}>
                  R{curRound} tree — weights being updated
                </div>
              </div>
            );
          })()}

          <div ref={canvasRef}
            style={{ width: "100%", height: "100%", overflow: "hidden", position: "relative",
              cursor: cursorGrabbing ? "grabbing" : "grab", zIndex: 1, background: "#080c14" }}
            onMouseDown={onMouseDown} onMouseMove={onMouseMove} onMouseUp={onMouseUp} onMouseLeave={onMouseUp}>
            {currentTree ? (
              <svg width={treeWidth} height={svgH}
                style={{ transform: `translate(${pan.x}px,${pan.y}px) scale(${zoom})`,
                  transformOrigin: "0 0", overflow: "visible", userSelect: "none" }}>
                <defs>
                  <filter id="lg" x="-50%" y="-50%" width="200%" height="200%">
                    <feDropShadow dx="0" dy="2" stdDeviation="3" floodColor={C.leafB} floodOpacity="0.3" />
                  </filter>
                  <filter id="ng" x="-50%" y="-50%" width="200%" height="200%">
                    <feDropShadow dx="0" dy="0" stdDeviation="5" floodColor={C.accent} floodOpacity="0.55" />
                  </filter>
                  <filter id="ng-p" x="-50%" y="-50%" width="200%" height="200%">
                    <feDropShadow dx="0" dy="0" stdDeviation="4" floodColor={C.orange} floodOpacity="0.6" />
                  </filter>
                  <filter id="path-glow" x="-50%" y="-50%" width="200%" height="200%">
                    <feDropShadow dx="0" dy="0" stdDeviation="5" floodColor={PATH_COLOR} floodOpacity="0.7" />
                  </filter>
                </defs>
                {allNodes.filter(n => n.type === "split").flatMap(n => [
                  { parent: n, child: n.left,  label: `≤${fmtThresh(n.threshold)}` },
                  { parent: n, child: n.right, label: `>${fmtThresh(n.threshold)}`  },
                ]).map(({ parent, child, label }) => (
                  <Edge key={`${parent.id}-${child.id}`}
                    p1={positions[parent.id]} p2={positions[child.id]}
                    visible={visibleSet.has(parent.id) && visibleSet.has(child.id)}
                    label={label}
                    onPath={samplePath.has(parent.id) && samplePath.has(child.id)}
                    sampleActive={!!selectedSample} />
                ))}
                {allNodes.map(n => (
                  <TreeNode key={n.id} node={n}
                    show={visibleSet.has(n.id) || (ts.nodeId === n.id && roundPhase === 1)}
                    phase={ts.nodeId === n.id && roundPhase === 1 ? ts.phase : 2}
                    pos={positions[n.id]} allClasses={allClasses}
                    onPath={samplePath.has(n.id)} sampleActive={!!selectedSample} />
                ))}
              </svg>
            ) : (
              <div style={{ display: "flex", alignItems: "center", justifyContent: "center",
                height: "100%", flexDirection: "column", gap: 8 }}>
                <div style={{ fontSize: 12, color: C.dim }}>{buildProgress ? "Training…" : "Press Grow or → to start"}</div>
              </div>
            )}

            {/* Navigation hint */}
            {selectedSample && (
              <button onClick={() => predSectionRef.current?.scrollIntoView({ behavior: "smooth", block: "start" })}
                style={{ position: "absolute", bottom: 10, right: 10, borderRadius: 20, border: `1px solid ${C.border}`,
                  background: C.panel, color: C.dim, fontSize: 10, padding: "5px 12px", cursor: "pointer",
                  fontFamily: "inherit", transition: "color 0.15s, border-color 0.15s, background 0.15s" }}
                onMouseEnter={e => { e.currentTarget.style.color = C.text; e.currentTarget.style.borderColor = C.dim; e.currentTarget.style.background = "#1a2235"; }}
                onMouseLeave={e => { e.currentTarget.style.color = C.dim; e.currentTarget.style.borderColor = C.border; e.currentTarget.style.background = C.panel; }}>
                ↓ Back to prediction
              </button>
            )}

            {/* Hint overlay — shown at initial state (root node visible or empty, before user steps) */}
            {!hintDismissed && roundData.length > 0 && roundPhase === 1 && ts.stepIdx <= 0 && !growing && !buildProgress && (
              <div style={{
                position: "absolute", inset: 0, display: "flex", flexDirection: "column",
                alignItems: "center", justifyContent: "center", pointerEvents: "none", zIndex: 5,
              }}>
                <div style={{
                  background: "rgba(10,14,23,0.82)",
                  border: "1px solid rgba(255,255,255,0.08)",
                  borderRadius: 16,
                  padding: "22px 32px",
                  maxWidth: 420, textAlign: "center",
                  backdropFilter: "blur(6px)",
                  boxShadow: "0 8px 40px rgba(0,0,0,0.6)",
                }}>
                  <div style={{ fontSize: 13, color: C.text, fontWeight: 600, marginBottom: 10, lineHeight: 1.5 }}>
                    Use <span style={{ color: C.accent, fontWeight: 700 }}>→</span> to step through the tree,
                    or press <span style={{ color: C.accent, fontWeight: 700 }}>▶ Grow</span> to watch it build automatically.
                  </div>
                  <div style={{ fontSize: 10.5, color: C.dim, lineHeight: 1.7 }}>
                    You can also change the dataset above to visualize on your own data.
                  </div>
                </div>
              </div>
            )}

            {/* Welcome overlay — shown while building (no roundData yet) */}
            {!buildProgress && roundPhase === -1 && roundData.length === 0 && (
              <div style={{ position: "absolute", inset: 0, display: "flex", alignItems: "center",
                justifyContent: "center", background: "rgba(10,14,23,0.6)", backdropFilter: "blur(2px)" }}>
                <div style={{ textAlign: "center", padding: "24px 32px", borderRadius: 16,
                  background: C.panel, boxShadow: "0 8px 32px rgba(0,0,0,0.5), inset 0 0 0 1px rgba(255,255,255,0.06)" }}>
                  <div style={{ fontSize: 13, fontWeight: 700, color: C.text, marginBottom: 6 }}>AdaBoost Visualizer</div>
                  <div style={{ fontSize: 10, color: C.dim, lineHeight: 1.7, marginBottom: 16 }}>
                    Press <span style={{ color: C.accent }}>Grow</span> to train sequentially,<br />
                    or <span style={{ color: C.accent }}>→</span> to step through one phase at a time.
                  </div>
                </div>
              </div>
            )}

            {/* Progress indicator */}
            {roundData.length > 0 && (
              <div style={{ position: "absolute", bottom: 8, left: 12, fontSize: 9, color: C.dimmer,
                fontFamily: "'JetBrains Mono',monospace", pointerEvents: "none" }}>
                {roundPhase >= 1 && roundPhase <= 1
                  ? `Step ${ts.stepIdx + 1}/${totalSteps}`
                  : roundPhase >= 0 ? PHASE_LABELS[roundPhase]?.text : ""}
              </div>
            )}
          </div>
        </div>

        {/* ── Calculations sidebar — draggable/collapsible, matching RF ── */}
        <div style={{
          width: calcExpanded ? calcPanelWidth : 28,
          flexShrink: 0,
          borderLeft: "1px solid rgba(255,255,255,0.05)",
          background: "#080c14",
          position: "relative",
          display: "flex",
          flexDirection: "column",
          overflow: "hidden",
        }}>
          {/* Drag handle / toggle */}
          <div
            onMouseDown={onCalcHandleMouseDown}
            title={calcExpanded ? "Drag to resize · Click to collapse" : "Click to expand"}
            style={{
              position: "absolute", left: 0, top: 0, bottom: 0,
              width: 10, zIndex: 5,
              cursor: calcExpanded ? "col-resize" : "pointer",
              display: "flex", alignItems: "center", justifyContent: "center",
              borderRight: "1px solid rgba(255,255,255,0.05)",
              background: "transparent",
              transition: "background 0.15s",
              userSelect: "none",
            }}
            onMouseEnter={e => { e.currentTarget.style.background = "rgba(255,255,255,0.05)"; }}
            onMouseLeave={e => { e.currentTarget.style.background = "transparent"; }}
          >
            {calcExpanded ? (
              <div style={{ display: "flex", flexDirection: "column", gap: 3, pointerEvents: "none" }}>
                {[0,1,2].map(i => (
                  <div key={i} style={{ width: 2, height: 2, borderRadius: "50%", background: C.dim, opacity: 0.5 }} />
                ))}
              </div>
            ) : (
              <span style={{ fontSize: 8, color: C.dim, pointerEvents: "none" }}>‹</span>
            )}
          </div>

          {/* Collapsed: rotated label */}
          {!calcExpanded && (
            <div style={{ position: "absolute", inset: 0, display: "flex",
              alignItems: "center", justifyContent: "center", pointerEvents: "none" }}>
              <span style={{ fontSize: 8.5, color: C.muted, fontWeight: 500,
                transform: "rotate(-90deg)", whiteSpace: "nowrap", letterSpacing: "0.08em" }}>
                Calculations
              </span>
            </div>
          )}

          {/* Expanded content */}
          {calcExpanded && (
            <div style={{ flex: 1, minHeight: 0, overflowY: "auto", overflowX: "hidden",
              paddingLeft: 12,
              scrollbarWidth: "thin", scrollbarColor: `${C.dimmer} transparent` }}>
              <div style={{ fontSize: 9, color: C.muted, fontWeight: 500, padding: "10px 10px 6px 6px" }}>Calculations</div>

              {/* Phase 1: Gini bar chart (same layout as RF) */}
              {roundPhase === 1 && currentNode && currentNode.type === "split" && (() => {
                const node = currentNode;
                const evals = node.allFeatureEvals ?? [];
                const candidateEvals = evals
                  .filter(e => (node.candidateIndices ?? []).includes(e.featureIndex))
                  .sort((a, b) => a.gini - b.gini);
                const maxGini = Math.max(...candidateEvals.map(e => e.gini), 0.001);
                return (
                  <div style={{ display: "flex", flexDirection: "column", gap: 6, paddingRight: 10 }}>
                    <div style={{ fontSize: 9, color: C.muted, paddingLeft: 6 }}>Node n={node.samples}</div>
                    {ts.phase >= 1 && (
                      <div style={{ borderTop: "1px solid rgba(255,255,255,0.05)", padding: "10px 10px 10px 6px" }}>
                        <div style={{ fontSize: 8.5, color: C.dim, marginBottom: 7 }}>
                          Gini per candidate <span style={{ color: C.muted }}>(shorter = better)</span>
                        </div>
                        <div style={{
                          display: "grid",
                          gridTemplateColumns: "auto minmax(20px, 1fr) 40px",
                          columnGap: 8,
                          rowGap: 5,
                          alignItems: "center",
                          overflow: "hidden",
                        }}>
                          {candidateEvals.map((ev, j) => {
                            const isChosen = ts.phase >= 2 && ev.featureIndex === node.featureIndex;
                            const barPct = maxGini > 0 ? (ev.gini / maxGini) * 100 : 50;
                            const rowColor = isChosen ? C.green : C.dim;
                            const fadeStyle = {
                              opacity: 0,
                              animation: `taxFadeIn 0.25s ease ${j * 0.06}s forwards`,
                            };
                            return [
                              /* Feature name cell */
                              <div key={`n-${ev.featureIndex}`} style={{
                                ...fadeStyle,
                                fontSize: 8.5, color: rowColor, fontWeight: isChosen ? 600 : 400,
                                whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis",
                                minWidth: 0, transition: "color .2s",
                              }} title={activeFeatures[ev.featureIndex]}>
                                {activeFeatures[ev.featureIndex]}
                              </div>,
                              /* Bar cell */
                              <div key={`b-${ev.featureIndex}`} style={{
                                ...fadeStyle,
                                height: 4, background: "rgba(255,255,255,0.05)",
                                borderRadius: 2, overflow: "hidden",
                              }}>
                                <div style={{
                                  height: "100%", width: `${barPct}%`,
                                  background: isChosen ? C.green : `${C.accent}88`,
                                  borderRadius: 2,
                                  transition: "width 0.4s ease-out, background 0.2s",
                                }} />
                              </div>,
                              /* Gini value cell */
                              <div key={`v-${ev.featureIndex}`} style={{
                                ...fadeStyle,
                                fontSize: 8, color: rowColor, fontFamily: "'JetBrains Mono',monospace",
                                textAlign: "right", whiteSpace: "nowrap", transition: "color .2s",
                              }}>
                                {ev.gini?.toFixed(3)}{isChosen && <span style={{ marginLeft: 3, fontSize: 7 }}>✓</span>}
                              </div>,
                            ];
                          })}
                        </div>
                        {ts.phase >= 2 && (
                          <div style={{ marginTop: 8, padding: "5px 7px", borderRadius: 7,
                            background: `${C.green}12`, border: `1px solid ${C.green}33`,
                            fontSize: 8.5, color: C.green }}>
                            ✓ Best: {activeFeatures[node.featureIndex]} ≤ {fmtThresh(node.threshold)}
                            <br /><span style={{ opacity: 0.75 }}>Weighted Gini = {node.gini?.toFixed(4)}</span>
                          </div>
                        )}
                      </div>
                    )}
                  </div>
                );
              })()}

              {/* Phase 2: Prediction summary */}
              {roundPhase === 2 && currentRoundData && (
                <div style={{ display: "flex", flexDirection: "column", gap: 8, paddingRight: 10, paddingLeft: 6 }}>
                  <div style={{ fontSize: 9, color: C.muted }}>Round {curRound + 1} predictions on training data:</div>
                  {(() => {
                    const correct = currentRoundData.misclassified.filter(m => !m).length;
                    const wrong   = currentRoundData.misclassified.filter(m => m).length;
                    const n       = activeData.length;
                    return (
                      <>
                        <div style={{ padding: "7px 10px", borderRadius: 10, background: `${C.green}14`,
                          border: `1px solid ${C.green}33`, fontSize: 10, color: C.green, fontFamily: "'JetBrains Mono',monospace" }}>
                          ✓ {correct}/{n} correct ({(correct/n*100).toFixed(1)}%)
                        </div>
                        <div style={{ padding: "7px 10px", borderRadius: 10, background: `${C.red}14`,
                          border: `1px solid ${C.red}33`, fontSize: 10, color: C.red, fontFamily: "'JetBrains Mono',monospace" }}>
                          ✗ {wrong}/{n} misclassified
                        </div>
                        <div style={{ fontSize: 8, color: C.dim, lineHeight: 1.6 }}>
                          Misclassified samples will have their weights increased in the next step.
                        </div>
                      </>
                    );
                  })()}
                </div>
              )}

              {/* Phase 3: Error + alpha */}
              {roundPhase >= 3 && currentRoundData && (
                <div style={{ display: "flex", flexDirection: "column", gap: 8, paddingRight: 10, paddingLeft: 6 }}>
                  <div style={{ padding: "8px 10px", borderRadius: 10, background: "rgba(255,255,255,0.03)",
                    border: "1px solid rgba(255,255,255,0.08)", fontSize: 9, fontFamily: "'JetBrains Mono',monospace", lineHeight: 1.85 }}>
                    <div style={{ color: C.orange }}>
                      ε = Σ wᵢ × 1[mᵢ] = <span style={{ fontWeight: 700 }}>{currentRoundData.error.toFixed(4)}</span>
                    </div>
                    <div style={{ color: C.dim, fontSize: 8, marginTop: 2 }}>weighted misclassification rate</div>
                  </div>
                  <div style={{ padding: "8px 10px", borderRadius: 10, background: `${C.purple}0d`,
                    border: `1px solid ${C.purple}33`, fontSize: 9, fontFamily: "'JetBrains Mono',monospace", lineHeight: 1.85 }}>
                    <div style={{ color: C.purple }}>α = ½ ln((1−ε)/ε)</div>
                    <div style={{ color: C.purple, fontWeight: 700 }}>
                      α = ½ ln({((1 - currentRoundData.error) / currentRoundData.error).toFixed(2)}) = {currentRoundData.alpha}
                    </div>
                    <div style={{ color: C.dim, fontSize: 8, marginTop: 2 }}>tree weight — higher α means better learner</div>
                  </div>
                </div>
              )}

              {/* Phase 4: Weight update formula */}
              {roundPhase === 4 && currentRoundData && (
                <div style={{ display: "flex", flexDirection: "column", gap: 6, paddingRight: 10, paddingLeft: 6 }}>
                  <div style={{ fontSize: 9, color: C.muted, marginBottom: 2 }}>Updating sample weights…</div>
                  <div style={{ padding: "7px 10px", borderRadius: 10, background: `${C.green}0d`,
                    border: `1px solid ${C.green}33`, fontSize: 8.5, fontFamily: "'JetBrains Mono',monospace", lineHeight: 1.9 }}>
                    <div style={{ color: C.green }}>Correct: wᵢ ← wᵢ × e^(−α)</div>
                    <div style={{ color: C.green, opacity: 0.7 }}>× {Math.exp(-currentRoundData.alpha).toFixed(4)}</div>
                  </div>
                  <div style={{ padding: "7px 10px", borderRadius: 10, background: `${C.red}0d`,
                    border: `1px solid ${C.red}33`, fontSize: 8.5, fontFamily: "'JetBrains Mono',monospace", lineHeight: 1.9 }}>
                    <div style={{ color: C.red }}>Misclassified: wᵢ ← wᵢ × e^(α)</div>
                    <div style={{ color: C.red, opacity: 0.7 }}>× {Math.exp(currentRoundData.alpha).toFixed(4)}</div>
                  </div>
                  <div style={{ fontSize: 8, color: C.dim }}>Then normalize so Σwᵢ = 1</div>
                </div>
              )}

              {/* Phase 5: Round summary */}
              {roundPhase === 5 && currentRoundData && (
                <div style={{ display: "flex", flexDirection: "column", gap: 8, paddingRight: 10, paddingLeft: 6 }}>
                  <div style={{ padding: "10px 12px", borderRadius: 12, background: `${C.green}0f`,
                    border: `1px solid ${C.green}33`, fontSize: 10, color: C.green, fontFamily: "'JetBrains Mono',monospace" }}>
                    ✔ Round {curRound + 1} complete
                    <div style={{ fontSize: 9, marginTop: 3, opacity: 0.8 }}>
                      α = {currentRoundData.alpha} · ε = {(currentRoundData.error * 100).toFixed(1)}%
                    </div>
                  </div>
                  {cumulativeAccuracy[curRound] !== undefined && (
                    <div style={{ padding: "7px 10px", borderRadius: 10, background: "rgba(255,255,255,0.03)",
                      border: "1px solid rgba(255,255,255,0.08)", fontSize: 9, color: C.text }}>
                      Ensemble accuracy: <span style={{ color: C.green, fontWeight: 700 }}>{cumulativeAccuracy[curRound]}%</span>
                      <div style={{ fontSize: 8, color: C.dim, marginTop: 2 }}>
                        on training data after {curRound + 1} round{curRound > 0 ? "s" : ""}
                      </div>
                    </div>
                  )}
                  {curRound < roundData.length - 1 && (
                    <div style={{ fontSize: 8.5, color: C.dim, textAlign: "center" }}>
                      → to continue to Round {curRound + 2}
                    </div>
                  )}
                </div>
              )}

              {/* Default / idle state */}
              {(roundPhase === -1 || (roundPhase === 1 && ts.phase === 0 && ts.stepIdx <= 0)) && (
                <div style={{ fontSize: 9, color: C.dimmer, lineHeight: 1.7, paddingRight: 10, paddingLeft: 6 }}>
                  {roundPhase === 1 ? (
                    <>
                      <div style={{ color: C.accent, fontWeight: 600, marginBottom: 4 }}>Round {curRound + 1} — Initial weights</div>
                      All {activeData.length} samples have equal weight:{" "}
                      <span style={{ fontFamily: "'JetBrains Mono',monospace", color: C.text }}>
                        1/{activeData.length} = {(1/activeData.length).toFixed(4)}
                      </span>
                      <div style={{ marginTop: 6 }}>Press → to start building the tree.</div>
                    </>
                  ) : "Build to start"}
                </div>
              )}
            </div>
          )}
        </div>
      </div>

      {/* ── Feature Pool — horizontal pills at bottom, matching RF ──────────── */}
      <div style={{ padding: "12px 16px 6px" }}>
        <div style={{
          background: "#0c1018", borderRadius: 14,
          boxShadow: "0 2px 16px rgba(0,0,0,0.35), inset 0 0 0 1px rgba(255,255,255,0.04)",
          padding: "14px 18px",
        }}>
          <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 10, flexWrap: "wrap" }}>
            <span style={{ fontSize: 9, color: C.muted, fontWeight: 400 }}>Feature pool</span>
            <span style={{ fontSize: 8.5 }}>
              <span style={{ color: C.dim }}>● not sampled</span>{"  "}
              <span style={{ color: `${C.accent}bb` }}>● candidate</span>{"  "}
              <span style={{ color: C.green }}>● chosen</span>
            </span>
          </div>
          <div style={{ display: "flex", flexWrap: "wrap", gap: 5, justifyContent: "center" }}>
            {activeFeatures.map((fname, fi) => {
              const isCandidate = fpCandidates.has(fi) && roundPhase === 1 && ts.phase >= 1;
              const isChosen    = fi === fpChosen && ts.phase >= 2 && roundPhase === 1;
              const ev          = fpEvals[fi];
              const showGini    = roundPhase === 1 && ts.phase >= 1 && ev && isCandidate;
              let bg, col, shd = "none";
              if (isChosen) {
                bg = `${C.green}22`; col = C.green; shd = `0 0 14px ${C.green}22`;
              } else if (isCandidate) {
                bg = `${C.accent}12`; col = `${C.accent}bb`;
              } else {
                bg = "rgba(255,255,255,0.03)"; col = C.dim;
              }
              return (
                <div key={fi} style={{
                  padding: "5px 11px", borderRadius: 10, fontSize: 10,
                  fontFamily: "'JetBrains Mono',monospace",
                  fontWeight: isChosen ? 600 : 400,
                  background: bg, color: col, boxShadow: shd,
                  transition: "all .3s ease-out", minWidth: 80, textAlign: "center",
                }}>
                  {fname}
                  {showGini && (
                    <div style={{ fontSize: 7.5, marginTop: 1, color: col }}>
                      G={ev.gini?.toFixed(3)}
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        </div>
      </div>

      {/* ── Step 2: Prediction ───────────────────────────────────────────────── */}
      <div ref={predSectionRef} style={{ borderTop: `1px solid ${C.border}`, padding: "0 16px 24px" }}>
        {/* Scroll hint */}
        <div style={{ height: 28, opacity: scrollHint ? 1 : 0, transition: "opacity 0.6s ease",
          pointerEvents: "none", textAlign: "center", paddingTop: 6 }}>
          <span style={{ fontSize: 10, color: C.accent, fontWeight: 600 }}>Scroll down to predict ↓</span>
        </div>

        {/* Step header */}
        <div style={{
          display: "flex", alignItems: "center", gap: 8, marginBottom: 10,
          padding: "10px 0 8px",
        }}>
          <span style={{ fontSize: 11, fontWeight: 700, color: PATH_COLOR, letterSpacing: "0.06em", flexShrink: 0 }}>Step 2</span>
          <span style={{ fontSize: 11, fontWeight: 700, color: C.text, letterSpacing: "0.04em", flexShrink: 0 }}>Prediction</span>
          <div style={{ flex: 1, height: 1, background: `linear-gradient(to right, ${PATH_COLOR}40, transparent)` }} />
          <span style={{ fontSize: 9, color: C.muted, flexShrink: 0 }}>{completedCount}/{roundData.length} rounds ready</span>
        </div>

        {/* Bordered content panel — matches RF's #0c1018 container */}
        <div style={{
          padding: "14px 16px", background: "#0c1018", borderRadius: 14,
          boxShadow: "0 2px 16px rgba(0,0,0,0.35), inset 0 0 0 1px rgba(255,255,255,0.04)",
        }}>
          {completedCount === 0 ? (
            <div style={{ fontSize: 10.5, color: C.dim, padding: "2px 0" }}>
              Complete at least one round to make predictions.
            </div>
          ) : (<>

            {/* ↑ View path in tree — top-right when sample is selected */}
            {selectedSample && (
              <div style={{ display: "flex", justifyContent: "flex-end", marginBottom: 8, marginTop: -2 }}>
                <button
                  onClick={() => canvasRef.current?.scrollIntoView({ behavior: "smooth", block: "start" })}
                  style={{
                    padding: "5px 13px", borderRadius: 20,
                    background: "rgba(255,255,255,0.05)",
                    border: "1px solid rgba(255,255,255,0.1)",
                    color: C.muted, fontSize: 10, fontWeight: 600,
                    cursor: "pointer", fontFamily: "inherit",
                    boxShadow: "0 2px 8px rgba(0,0,0,0.3)",
                    transition: "color 0.15s, border-color 0.15s, background 0.15s",
                  }}
                  onMouseEnter={e => { e.currentTarget.style.color = C.text; e.currentTarget.style.borderColor = "rgba(255,255,255,0.22)"; e.currentTarget.style.background = "rgba(255,255,255,0.09)"; }}
                  onMouseLeave={e => { e.currentTarget.style.color = C.muted; e.currentTarget.style.borderColor = "rgba(255,255,255,0.1)"; e.currentTarget.style.background = "rgba(255,255,255,0.05)"; }}
                >
                  ↑ View path in tree
                </button>
              </div>
            )}

            {/* ── Sample selector ── */}
            <div style={{ display: "flex", alignItems: "center", gap: 12, marginBottom: 16 }}>
              <div style={{
                position: "relative", display: "flex", alignItems: "center",
                borderRadius: 16, background: "rgba(255,255,255,0.04)",
                boxShadow: "0 8px 32px rgba(0,0,0,0.4), 0 0 0 1px rgba(255,255,255,0.08)",
                maxWidth: 500, flex: 1,
              }}>
                <div style={{ padding: "0 14px 0 18px", color: C.dim, fontSize: 16, flexShrink: 0 }}>⌕</div>
                <select
                  value={safeSampleIdx ?? ""}
                  onChange={e => setSelectedSampleIdx(e.target.value === "" ? null : Number(e.target.value))}
                  style={{
                    flex: 1, minWidth: 0, background: "transparent", border: "none",
                    color: C.text, fontSize: 12, padding: "12px 8px 12px 0",
                    outline: "none", fontFamily: "inherit", cursor: "pointer",
                  }}
                >
                  <option value="">Select a sample to predict…</option>
                  {activeData.map((row, i) => (
                    <option key={i} value={i}>{formatSampleLabel(row, activeFeatures, i)}</option>
                  ))}
                </select>
              </div>
              <button
                onClick={() => setSelectedSampleIdx(Math.floor(Math.random() * activeData.length))}
                style={{
                  flexShrink: 0, padding: "6px 14px", borderRadius: 20, border: "none",
                  background: `${C.accent}22`, color: C.accent, fontSize: 11,
                  fontFamily: "inherit", cursor: "pointer", fontWeight: 600,
                  transition: "background 0.15s, box-shadow 0.15s",
                }}
                onMouseEnter={e => { e.currentTarget.style.background = `${C.accent}38`; e.currentTarget.style.boxShadow = "0 0 10px rgba(245,158,11,0.2)"; }}
                onMouseLeave={e => { e.currentTarget.style.background = `${C.accent}22`; e.currentTarget.style.boxShadow = "none"; }}
              >
                Random
              </button>
            </div>

            {/* ── No sample selected ── */}
            {!selectedSample && (
              <div style={{ fontSize: 10.5, color: C.dim, paddingBottom: 4, textAlign: "center", lineHeight: 1.7 }}>
                Select a sample above to see how each round votes and follow its path through the tree.
              </div>
            )}

            {/* ── Ensemble card + round vote cards ── */}
            {selectedSample && (() => {
              // Ensemble card (left, 210px) — mirrors RF's classification ensemble card
              const ensembleCard = ensembleScores ? (() => {
                const predColor = classColor(ensemblePrediction, allClasses);
                const truLabel  = selectedSample[activeTargetCol];
                const isCorrect = ensemblePrediction === truLabel;
                return (
                  <div style={{
                    flexShrink: 0, width: 210, alignSelf: "stretch",
                    padding: "16px 16px", borderRadius: 14,
                    background: `linear-gradient(135deg, ${predColor}12, rgba(10,14,23,0.6))`,
                    boxShadow: `0 0 0 1.5px ${predColor}55, 0 4px 24px ${predColor}18, inset 0 1px 0 ${predColor}22`,
                    display: "flex", flexDirection: "column", justifyContent: "center",
                  }}>
                    <div style={{ fontSize: 8.5, color: C.dim, fontWeight: 500, marginBottom: 5, letterSpacing: "0.06em", textTransform: "uppercase" }}>
                      Ensemble · {completedCount} round{completedCount !== 1 ? "s" : ""}
                    </div>
                    <div style={{ fontSize: 22, fontWeight: 800, color: predColor, lineHeight: 1, marginBottom: 10 }}>
                      {ensemblePrediction}
                    </div>
                    {/* Alpha-sum bar per class */}
                    <div style={{ marginBottom: 8 }}>
                      <div style={{ height: 5, borderRadius: 3, overflow: "hidden", background: "rgba(0,0,0,0.3)", display: "flex", marginBottom: 5 }}>
                        {Object.entries(ensembleScores).sort((a, b) => b[1] - a[1]).map(([cls, score]) => {
                          const col = classColor(cls, allClasses);
                          const pct = totalAlpha > 0 ? (score / totalAlpha) * 100 : 0;
                          return pct > 0 ? <div key={cls} style={{ width: `${pct}%`, background: col, transition: "width .3s" }} /> : null;
                        })}
                      </div>
                      <div style={{ display: "flex", flexDirection: "column", gap: 3 }}>
                        {Object.entries(ensembleScores).sort((a, b) => b[1] - a[1]).map(([cls, score]) => {
                          const col = classColor(cls, allClasses);
                          if (score === 0) return null;
                          return (
                            <span key={cls} style={{ fontSize: 8.5, color: C.muted }}>
                              <strong style={{ color: col, fontFamily: "'JetBrains Mono',monospace" }}>{score.toFixed(2)}</strong>
                              <span style={{ color: C.dimmer }}>/{totalAlpha.toFixed(2)} α · </span>
                              <span style={{ color: col }}>{cls}</span>
                            </span>
                          );
                        })}
                      </div>
                    </div>
                    <div style={{ display: "flex", gap: 10, flexWrap: "wrap", alignItems: "center", paddingTop: 8, borderTop: "1px solid rgba(255,255,255,0.06)" }}>
                      <div style={{ display: "flex", alignItems: "center", gap: 5 }}>
                        <span style={{ fontSize: 8.5, color: C.muted, textTransform: "uppercase", letterSpacing: "0.05em" }}>True</span>
                        <span style={{ fontSize: 12, fontWeight: 700, color: classColor(String(truLabel), allClasses) }}>{String(truLabel)}</span>
                      </div>
                      <span style={{ fontSize: 12, fontWeight: 800, color: isCorrect ? C.green : C.red }}>
                        {isCorrect ? "✓ Correct" : "✗ Wrong"}
                      </span>
                    </div>
                  </div>
                );
              })() : (
                <div style={{
                  flexShrink: 0, width: 210, alignSelf: "stretch",
                  padding: "16px 16px", borderRadius: 14,
                  background: "rgba(255,255,255,0.02)",
                  boxShadow: "inset 0 0 0 1px rgba(255,255,255,0.06)",
                  display: "flex", flexDirection: "column", justifyContent: "center",
                }}>
                  <div style={{ fontSize: 8.5, color: C.dim, textTransform: "uppercase", letterSpacing: "0.06em", marginBottom: 6 }}>Ensemble</div>
                  <div style={{ fontSize: 12, color: C.dimmer }}>Complete rounds to see ensemble prediction.</div>
                </div>
              );

              // Round vote cards (right side) — fixed 46px height, matching RF's tree cards
              const roundCards = (
                <div style={{ display: "flex", gap: 5, flexWrap: "wrap", alignItems: "center" }}>
                  {sampleRoundPreds.map(({ round, alpha: a, pred }) => {
                    const col      = classColor(pred, allClasses);
                    const abbrev   = pred && pred.length > 9 ? pred.slice(0, 8) + "…" : pred;
                    const isActive = round - 1 === curRound;
                    return (
                      <button key={round}
                        onClick={() => {
                          setCurRound(round - 1);
                          setRoundPhase(5);
                          setWeightDisplayMode("output");
                        }}
                        style={{
                          minWidth: 46, height: 46, borderRadius: 9, border: "none",
                          display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center",
                          padding: "0 7px", cursor: "pointer",
                          background: isActive ? `${col}22` : `${col}0d`,
                          boxShadow: isActive
                            ? `inset 0 0 0 2px ${col}55`
                            : `inset 0 0 0 1px ${col}33`,
                          transition: "box-shadow .2s, background .2s",
                          fontFamily: "'JetBrains Mono',monospace",
                        }}
                        onMouseEnter={e => { e.currentTarget.style.background = `${col}28`; }}
                        onMouseLeave={e => { e.currentTarget.style.background = isActive ? `${col}22` : `${col}0d`; }}
                      >
                        <div style={{ fontSize: 7, color: isActive ? C.text : C.dim }}>R{round}</div>
                        <div style={{ fontSize: 9, fontWeight: 700, color: col, whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis", maxWidth: 40 }}>
                          {abbrev}
                        </div>
                        <div style={{ fontSize: 7, color: `${col}99`, fontFamily: "'JetBrains Mono',monospace" }}>
                          α={a}
                        </div>
                      </button>
                    );
                  })}
                </div>
              );

              return (
                <div style={{ display: "flex", gap: 12, alignItems: "center", marginBottom: 14 }}>
                  {ensembleCard}
                  {roundCards}
                </div>
              );
            })()}

            {/* ── Metrics cards — always visible when rounds completed ── */}
            <div style={{ display: "flex", gap: 10, marginTop: selectedSample ? 4 : 0 }}>
              <div style={{
                flex: 1, padding: "12px 14px", borderRadius: 12,
                background: "rgba(255,255,255,0.03)",
                boxShadow: "inset 0 0 0 1px rgba(255,255,255,0.07)",
              }}>
                <div style={{ fontSize: 8, color: C.dim, fontWeight: 600, letterSpacing: "0.08em", textTransform: "uppercase", marginBottom: 6 }}>Round {completedCount} Error</div>
                <div style={{ fontSize: 9, color: C.muted, marginBottom: 2 }}>Weighted misclassification</div>
                <div style={{ fontSize: 20, fontWeight: 800, color: C.text, fontFamily: "'JetBrains Mono',monospace" }}>
                  {(roundData[completedCount - 1]?.error * 100).toFixed(1)}%
                </div>
              </div>
              <div style={{
                flex: 1, padding: "12px 14px", borderRadius: 12,
                background: `${C.green}0d`,
                boxShadow: `inset 0 0 0 1px ${C.green}44`,
              }}>
                <div style={{ fontSize: 8, color: C.green, fontWeight: 600, letterSpacing: "0.08em", textTransform: "uppercase", marginBottom: 6 }}>Ensemble Accuracy</div>
                <div style={{ fontSize: 9, color: C.muted, marginBottom: 2 }}>Training · {completedCount} round{completedCount !== 1 ? "s" : ""}</div>
                <div style={{ fontSize: 20, fontWeight: 800, color: C.green, fontFamily: "'JetBrains Mono',monospace" }}>
                  {cumulativeAccuracy[completedCount - 1]}%
                </div>
              </div>
            </div>

          </>)}
        </div>
      </div>

      {/* Locked max-features tooltip */}
      {lockedTip && (
        <div style={{
          position: "fixed", left: lockedTip.left, top: lockedTip.top,
          width: 240, padding: "10px 13px", borderRadius: 10,
          background: "#1a2235", boxShadow: "0 8px 32px rgba(0,0,0,0.7), inset 0 0 0 1px rgba(255,255,255,0.12)",
          fontSize: 10.5, color: C.text, lineHeight: 1.65, zIndex: 9999, pointerEvents: "none",
        }}>
          AdaBoost evaluates all features — feature subsampling is a Random Forest concept.
        </div>
      )}

      {/* Reset button tooltip */}
      {resetTooltip && (
        <div style={{
          position: "fixed",
          left: resetTooltip.left,
          top: resetTooltip.top,
          zIndex: 9999,
          background: "#1a2235",
          borderRadius: 7,
          padding: "5px 10px",
          fontSize: 9,
          color: C.dim,
          boxShadow: "0 4px 16px rgba(0,0,0,0.5), inset 0 0 0 1px rgba(255,255,255,0.07)",
          pointerEvents: "none",
          whiteSpace: "nowrap",
        }}>
          Click to reset current round · Double-click to reset all
        </div>
      )}
    </div>
  );
}
