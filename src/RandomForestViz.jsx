import { useState, useEffect, useLayoutEffect, useRef, useCallback, useMemo, useImperativeHandle } from "react";
import { createPortal } from "react-dom";
import { useLocation } from "react-router-dom";
import Papa from "papaparse";
import { C } from "./theme";
import GlobalHeader from "./GlobalHeader";
import { heartData, heartMeta } from "./data/heartDisease";
import { musicData, musicMeta } from "./data/musicData";
import { salaryData, salaryMeta } from "./data/salary";
import { predictRow } from "./cartAlgorithm";
import TreeWorker from "./treeWorker.js?worker";
import DataModal from "./DataModal";
import { NA_VALS, detectNAs, detectTaskType } from "./dataUtils";
import { classColor, flattenNodes, getTreePrediction, fmtThresh, LEAF_SPACING, X_PAD, Y_GAP, computeTreeWidth, computePositions, formatRegVal, getSamplePath, formatSampleLabel, PATH_COLOR, Edge, TreeNode } from "./TreeComponents";
import { tooltipPosition } from "./tooltipUtils";
import { useTutorial } from "./Tutorial";

// ─── Feature-subset options ────────────────────────────────────────────────────
const FEATURE_SUBSET_OPTIONS = {
  sqrt: { label: "√p (sqrt)", fn: (p) => Math.max(1, Math.round(Math.sqrt(p))) },
  log2: { label: "log₂(p)",  fn: (p) => Math.max(1, Math.round(Math.log2(p))) },
  half: { label: "p/2",       fn: (p) => Math.max(1, Math.round(p / 2)) },
  all:  { label: "p (all)",   fn: (p) => p },
};

const EMPTY_TS = { visibleIds: [], nodeId: null, phase: 0, stepIdx: -1 };

// ─── Interaction hint tooltip (shown via ? button in header) ──────────────────
const ALGO_DESC = {
  "decision-tree": [
    "A single tree learned directly from data.",
    "At each node, finds the feature + threshold that minimizes Gini impurity.",
    "Fully interpretable — the foundation of all ensemble methods.",
  ],
  "bagging": [
    "Bootstrap Aggregating — trains many trees independently in parallel.",
    "Each tree is fit on a different bootstrap resample of the training data.",
    "Averaging predictions reduces variance without increasing bias.",
  ],
  "random-forest": [
    "Extends Bagging by also randomizing the feature set at each split.",
    "Each split considers only √n candidate features — decorrelates the trees.",
    "High accuracy, robust to overfitting, handles mixed feature types well.",
  ],
};

function AlgoTooltip({ mode }) {
  const [pos, setPos] = useState(null);
  const btnRef = useRef(null);
  const lines = ALGO_DESC[mode] ?? [];
  const show = () => {
    const r = btnRef.current?.getBoundingClientRect();
    if (r) setPos(tooltipPosition(r, { prefer: 'below', gap: 8, width: 280, height: 80 }));
  };
  return (
    <span style={{ display: "inline-flex", alignItems: "center", marginLeft: 8 }}>
      <span
        ref={btnRef}
        onMouseEnter={show}
        onMouseLeave={() => setPos(null)}
        style={{
          cursor: "help", fontSize: 10, color: C.dimmer,
          border: `1px solid ${C.border}`, borderRadius: "50%",
          width: 16, height: 16, display: "inline-flex",
          alignItems: "center", justifyContent: "center",
          lineHeight: 1, userSelect: "none",
          transition: "color 0.15s, border-color 0.15s",
        }}
        onMouseEnterCapture={e => { e.currentTarget.style.color = C.dim; e.currentTarget.style.borderColor = C.dim; }}
        onMouseLeaveCapture={e => { e.currentTarget.style.color = C.dimmer; e.currentTarget.style.borderColor = C.border; }}
      >?</span>
      {pos && createPortal(
        <div style={{
          position: "fixed", left: pos.left, top: pos.top,
          zIndex: 9999,
          background: "#141c2e", border: `1px solid rgba(255,255,255,0.08)`,
          borderRadius: 10, padding: "12px 16px",
          fontSize: 10.5, color: C.dim, lineHeight: 1.75,
          maxWidth: 280, pointerEvents: "none",
          boxShadow: "0 8px 32px rgba(0,0,0,0.6)",
        }}>
          {lines.map((l, i) => <div key={i}>{l}</div>)}
        </div>,
        document.body
      )}
    </span>
  );
}

// ─── Main component ────────────────────────────────────────────────────────────
export default function RandomForestViz({ mode = "random-forest", tutorialRef = null }) {
  const lockedNEstimators = mode === "decision-tree" ? 1 : null;
  const lockedMaxFeatures = (mode === "decision-tree" || mode === "bagging") ? "all" : null;

  // ── Dataset state ──────────────────────────────────────────────────────────
  const [builtinDataset, setBuiltinDataset] = useState("heart"); // "heart" | "music" | "salary"
  const [customDataset, setCustomDataset] = useState(null);
  const fileInputRef = useRef(null);
  const [csvModal, setCsvModal]           = useState(null);
  const [dragOver, setDragOver]           = useState(false);
  const [selectedSampleIdx, setSelectedSampleIdx] = useState(null);
  const [oobTooltipVisible, setOobTooltipVisible] = useState(null); // { top, left } | null

  const builtinMeta     = builtinDataset === "music" ? musicMeta : builtinDataset === "salary" ? salaryMeta : heartMeta;
  const builtinData     = builtinDataset === "music" ? musicData : builtinDataset === "salary" ? salaryData : heartData;
  const builtinTaskType = builtinDataset === "salary" ? "regression" : "classification";
  const activeData      = customDataset?.data        ?? builtinData;
  const activeFeatures  = customDataset?.features    ?? builtinMeta.features;
  const activeTargetCol = customDataset?.targetCol   ?? builtinMeta.targetCol;
  const classLabels     = customDataset?.classLabels ?? builtinMeta.targetLabels;
  const activeTaskType  = customDataset?.taskType    ?? builtinTaskType;

  // ── Hyperparameter state ───────────────────────────────────────────────────
  const [maxDepth, setMaxDepth]         = useState(3);
  const [maxDepthStr, setMaxDepthStr]   = useState("3");
  const [featureSubset, setFeatureSubset] = useState(lockedMaxFeatures ?? "sqrt");
  const [nEstimators, setNEstimators]   = useState(lockedNEstimators ?? 10);
  const [nEstimatorsStr, setNEstimatorsStr] = useState(String(lockedNEstimators ?? 10));
  const [trees, setTrees]               = useState([]);
  const [bootstrapInfo, setBootstrapInfo] = useState([]);
  const [curTree, setCurTree]           = useState(0);
  const [treeStates, setTreeStates]     = useState({});
  const [growing, setGrowing]           = useState(false);
  const [speed, setSpeed]               = useState(1);
  const [buildProgress, setBuildProgress] = useState(null); // null | { done, total }
  const location = useLocation();
  const growRef      = useRef(false);
  const cancelRef    = useRef(false);
  const workerRef    = useRef(null);
  const liveStepRef  = useRef({ treeIdx: 0, stepIdx: -1 }); // tracks exact position during animation
  const pausedAtRef  = useRef(null);                          // set on Stop, cleared on next Grow
  const resetClickRef = useRef(null);                         // timer for single vs double click on Reset

  const [tabTooltip, setTabTooltip]           = useState(null); // { x, y }
  const [resetTooltip, setResetTooltip]       = useState(null); // { x, y }
  const [dragRange, setDragRange]             = useState(null); // { start, end } | null
  const [hintDismissed, setHintDismissed]     = useState(false);
  // Incremented by resetAndSetParams to guarantee the tutorialRebuild effect fires
  // even when hyperparams don't change (buildForest would be the same ref).
  const [tutorialBuildTick, setTutorialBuildTick] = useState(0);
  const [scrollHint, setScrollHint]           = useState(false); // "Scroll down to predict ↓" hint
  const [lockedParamTooltip, setLockedParamTooltip] = useState(null); // { key, x, y }
  const [rfDropOpen,  setRfDropOpen]  = useState(false);
  const [rfDropPos,   setRfDropPos]   = useState(null); // { top, left } for portaled menu
  const rfDropRef  = useRef(null);
  const rfMenuRef  = useRef(null);
  const dragRef    = useRef({ active: false, startIdx: null, endIdx: null, moved: false });
  const tabRefs    = useRef([]);
  const tabScrollRef = useRef(null);
  const predSectionRef = useRef(null);

  const [calcExpanded, setCalcExpanded]     = useState(true);
  const [calcPanelWidth, setCalcPanelWidth] = useState(220);
  const [zoom, setZoom]                     = useState(1);
  const [pan, setPan]                       = useState({ x: 0, y: 0 });
  const [cursorGrabbing, setCursorGrabbing] = useState(false);
  const isDragging = useRef(false);
  const dragStart  = useRef({ x: 0, y: 0, px: 0, py: 0 });
  const canvasRef  = useRef(null);
  // Mutable refs so the wheel handler (registered once, deps=[]) always reads live values.
  const zoomLive = useRef(1);
  const panLive  = useRef({ x: 0, y: 0 });
  zoomLive.current = zoom;
  panLive.current  = pan;

  const subsetSize = FEATURE_SUBSET_OPTIONS[featureSubset].fn(activeFeatures.length);

  // ── Tutorial integration ───────────────────────────────────────────────────
  const tutorial = useTutorial();
  // Ref flag: set by resetAndSetParams() so the effect below triggers a rebuild
  // once the new state values have propagated to the buildForestWithData closure.
  const tutorialRebuildRef = useRef(false);

  // Inject design-system keyframes once
  useEffect(() => {
    const id = "rfv-keyframes";
    if (document.getElementById(id)) return;
    const s = document.createElement("style");
    s.id = id;
    s.textContent = `
      @keyframes growPulse {
        0%, 100% { box-shadow: 0 0 0 0 rgba(245,158,11,0); }
        50%       { box-shadow: 0 0 0 6px rgba(245,158,11,0.18); }
      }
      @keyframes nodeIn {
        from { opacity: 0; transform: scale(0.94); }
        to   { opacity: 1; transform: scale(1); }
      }
      @keyframes taxFadeIn {
        from { opacity: 0; }
        to   { opacity: 1; }
      }
      @keyframes fadeInUp {
        from { opacity: 0; transform: translateX(-50%) translateY(4px); }
        to   { opacity: 1; transform: translateX(-50%) translateY(0); }
      }
      .rf-ds-drop-menu { position: fixed; z-index: 99999; min-width: 230px;
        background: #1a2235; border: 1px solid rgba(255,255,255,0.12); border-radius: 8px; padding: 4px;
        box-shadow: 0 8px 32px rgba(0,0,0,0.6); }
      .rf-ds-opt { display: flex; align-items: center; gap: 6px; width: 100%; padding: 7px 10px; border-radius: 5px;
        border: none; background: none; font-family: 'Inter', system-ui, sans-serif; font-size: 11px;
        text-align: left; cursor: pointer; white-space: nowrap; color: #e2e8f0; transition: background 0.12s; box-sizing: border-box; }
      .rf-ds-opt:hover { background: rgba(255,255,255,0.08); }
      .rf-ds-opt[data-active="true"] { color: #f59e0b; }
    `;
    document.head.appendChild(s);
    return () => document.getElementById(id)?.remove();
  }, []);

  // Close dataset dropdown on outside click
  useEffect(() => {
    if (!rfDropOpen) return;
    const handler = (e) => {
      const inTrigger = rfDropRef.current?.contains(e.target);
      const inMenu    = rfMenuRef.current?.contains(e.target);
      if (!inTrigger && !inMenu) setRfDropOpen(false);
    };
    document.addEventListener("mousedown", handler);
    return () => document.removeEventListener("mousedown", handler);
  }, [rfDropOpen]);

  const getTS = (idx) => treeStates[idx] || EMPTY_TS;

  const setTS = useCallback((idx, patch) => {
    setTreeStates(prev => {
      const old = prev[idx] || EMPTY_TS;
      return { ...prev, [idx]: { ...old, ...patch } };
    });
  }, []);

  // ── Forest building ────────────────────────────────────────────────────────
  // Accepts data explicitly so it can be called right after state updates.
  // Uses a Web Worker to keep the UI responsive; posts each tree back individually.
  const buildForestWithData = useCallback((data, features, targetCol, taskType = "classification") => {
    cancelRef.current   = true;
    growRef.current     = false;
    pausedAtRef.current = null;
    setGrowing(false);

    if (workerRef.current) {
      workerRef.current.terminate();
      workerRef.current = null;
    }

    const activeS = lockedMaxFeatures ?? featureSubset;
    const subSize = FEATURE_SUBSET_OPTIONS[activeS].fn(features.length);
    const newTrees = new Array(nEstimators).fill(null);
    const newBInfo = new Array(nEstimators).fill(null);

    setTrees([...newTrees]);
    setBootstrapInfo([...newBInfo]);
    setCurTree(0);
    setTreeStates({});
    setZoom(1);
    setBuildProgress({ done: 0, total: nEstimators });
    setHintDismissed(false);
    setSelectedSampleIdx(null);

    const worker = new TreeWorker();
    workerRef.current = worker;

    worker.onmessage = ({ data: msg }) => {
      if (msg.type === "tree") {
        const { idx, tree, bInfo } = msg;
        newTrees[idx] = tree;
        newBInfo[idx] = bInfo;
        setTrees([...newTrees]);
        setBootstrapInfo([...newBInfo]);
        setBuildProgress(prev => prev ? { ...prev, done: prev.done + 1 } : null);
      } else if (msg.type === "done") {
        setBuildProgress(null);
        workerRef.current = null;
      }
    };

    worker.onerror = (err) => {
      console.error("Tree worker error:", err);
      setBuildProgress(null);
      workerRef.current = null;
    };

    worker.postMessage({ data, features, targetCol, maxDepth, subSize, nEstimators, mode: taskType });
  }, [maxDepth, featureSubset, nEstimators, lockedMaxFeatures]);

  const buildForest = useCallback(() => {
    buildForestWithData(activeData, activeFeatures, activeTargetCol, activeTaskType);
  }, [buildForestWithData, activeData, activeFeatures, activeTargetCol, activeTaskType]);

  const switchToBuiltin = useCallback((key) => {
    if (growRef.current) { cancelRef.current = true; growRef.current = false; setGrowing(false); }
    setBuiltinDataset(key);
    setCustomDataset(null);
    setSelectedSampleIdx(null);
    const d = key === "music" ? musicData : key === "salary" ? salaryData : heartData;
    const m = key === "music" ? musicMeta : key === "salary" ? salaryMeta : heartMeta;
    const t = key === "salary" ? "regression" : "classification";
    buildForestWithData(d, m.features, m.targetCol, t);
  }, [buildForestWithData]);

  useEffect(() => {
    const pending = location.state?.pendingCSV;
    if (pending) {
      // A CSV was dropped on the landing page — open the DataModal automatically
      const { data: rawRows, meta } = Papa.parse(pending.content, { header: true, skipEmptyLines: true });
      if (rawRows.length) {
        const headers    = meta.fields;
        const naStats    = detectNAs(rawRows, headers);
        const initTarget = headers[headers.length - 1];
        const modal = {
          fileName: pending.name, rawRows, headers, naStats,
          selectedTarget: initTarget,
          naStrategy: "drop", sampleMode: Math.min(1000, rawRows.length),
          selectedColumns: headers,
          taskType: detectTaskType(rawRows, initTarget),
        };
        setTimeout(() => setCsvModal(modal), 0);
        return;
      }
    }
    buildForest(); // eslint-disable-line react-hooks/set-state-in-effect
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  // ── Step / animation helpers ───────────────────────────────────────────────
  const getSteps = useCallback((treeIdx) => {
    const tree = trees[treeIdx];
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
  }, [trees]);

  const goToStep = useCallback((treeIdx, targetIdx) => {
    const steps = getSteps(treeIdx);
    if (!steps.length) return;
    if (targetIdx < -1) targetIdx = -1;
    if (targetIdx >= steps.length) targetIdx = steps.length - 1;
    if (targetIdx === -1) { setTS(treeIdx, { visibleIds: [], nodeId: null, phase: 0, stepIdx: -1 }); return; }
    const vis = [];
    for (let i = 0; i <= targetIdx; i++) {
      if (steps[i].commit && !vis.includes(steps[i].nodeId)) vis.push(steps[i].nodeId);
    }
    const s = steps[targetIdx];
    setTS(treeIdx, { visibleIds: vis, nodeId: s.nodeId, phase: s.phase, stepIdx: targetIdx });
  }, [getSteps, setTS]);

  const autoGrowAll = useCallback(async () => {
    if (growRef.current) return;
    growRef.current   = true;
    cancelRef.current = false;
    setGrowing(true);

    const resume       = pausedAtRef.current;
    pausedAtRef.current = null;
    const startTreeIdx  = resume?.treeIdx ?? 0;

    for (let treeIdx = startTreeIdx; treeIdx < trees.length; treeIdx++) {
      if (cancelRef.current) break;
      if (!trees[treeIdx]) continue;

      setCurTree(treeIdx);
      liveStepRef.current = { treeIdx, stepIdx: -1 };

      const isResumedTree = treeIdx === startTreeIdx && resume !== null;
      if (!isResumedTree) {
        setTS(treeIdx, EMPTY_TS);
        await new Promise(r => setTimeout(r, 80));
      }

      const steps    = getSteps(treeIdx);
      const fromStep = isResumedTree ? (resume.stepIdx + 1) : 0;

      for (let i = fromStep; i < steps.length; i++) {
        if (cancelRef.current) break;
        liveStepRef.current = { treeIdx, stepIdx: i };
        const vis = [];
        for (let j = 0; j <= i; j++) {
          if (steps[j].commit && !vis.includes(steps[j].nodeId)) vis.push(steps[j].nodeId);
        }
        setTS(treeIdx, { visibleIds: vis, nodeId: steps[i].nodeId, phase: steps[i].phase, stepIdx: i });
        await new Promise(r => setTimeout(r, 200 / speed));
      }

      if (!cancelRef.current && treeIdx < trees.length - 1) {
        await new Promise(r => setTimeout(r, 280));
      }
    }

    growRef.current = false;
    setGrowing(false);
  }, [trees, getSteps, setTS, speed]);

  const instantComplete = useCallback((treeIdx) => {
    const steps = getSteps(treeIdx);
    if (!steps.length) return;
    const lastIdx = steps.length - 1;
    const vis = [];
    for (let i = 0; i <= lastIdx; i++) {
      if (steps[i].commit && !vis.includes(steps[i].nodeId)) vis.push(steps[i].nodeId);
    }
    const s = steps[lastIdx];
    setTS(treeIdx, { visibleIds: vis, nodeId: s.nodeId, phase: s.phase, stepIdx: lastIdx });
  }, [getSteps, setTS]);

  const growAllInstant = useCallback(() => {
    if (growing) return;
    for (let i = 0; i < trees.length; i++) instantComplete(i);
  }, [growing, trees, instantComplete]);

  // Expose imperative methods to the Tutorial overlay via tutorialRef
  // (placed here, after all referenced callbacks are declared — avoids TDZ)
  useImperativeHandle(tutorialRef, () => ({
    resetAndSetParams({ maxDepth: md, featureSubset: fs, nEstimators: ne, speed: sp }) {
      if (growRef.current) { cancelRef.current = true; growRef.current = false; setGrowing(false); }
      setMaxDepth(md);   setMaxDepthStr(String(md));
      setFeatureSubset(fs);
      setNEstimators(ne); setNEstimatorsStr(String(ne));
      setSpeed(sp);
      setTrees([]); setBootstrapInfo([]); setTreeStates({}); setCurTree(0);
      setSelectedSampleIdx(null); setHintDismissed(false);
      // Always increment the tick so the rebuild effect fires even when
      // the new hyperparams are identical to the current ones (buildForest
      // would not change refs, so the effect would never fire without this).
      tutorialRebuildRef.current = true;
      setTutorialBuildTick(t => t + 1);
    },
    completeAll()        { growAllInstant(); },
    stepOnce() {
      // Jump to the next committed step (phase 2 of the current node) so the user
      // sees the feature/threshold revealed and the Feature Pool fully colored in
      // one key press, rather than landing on the intermediate phase-1 "?" state.
      const steps = getSteps(curTree);
      const currentIdx = getTS(curTree).stepIdx;
      let nextIdx = currentIdx + 1;
      while (nextIdx < steps.length && !steps[nextIdx].commit) nextIdx++;
      if (nextIdx < steps.length) goToStep(curTree, nextIdx);
    },
    selectRandomSample() { setSelectedSampleIdx(Math.floor(Math.random() * activeData.length)); },
    scrollToPrediction() { predSectionRef.current?.scrollIntoView({ behavior: "smooth", block: "start" }); },
  }), [growAllInstant, activeData.length, curTree, getTS, goToStep, getSteps]);

  // Rebuild after resetAndSetParams. Uses tutorialBuildTick so this fires even when
  // buildForest didn't change (hyperparams already matched the target values).
  // tutorialRebuildRef guards against spurious fires from buildForest changing
  // independently (e.g. user changes a param control after the tutorial rebuild).
  useEffect(() => {
    if (!tutorialRebuildRef.current) return;
    tutorialRebuildRef.current = false;
    buildForest();
  }, [tutorialBuildTick, buildForest]); // eslint-disable-line react-hooks/exhaustive-deps

  // ── Keyboard handler ───────────────────────────────────────────────────────
  useEffect(() => {
    const h = (e) => {
      if (growing || buildProgress) return;
      if (e.key === "ArrowRight" || e.key === "ArrowDown") {
        e.preventDefault();
        setHintDismissed(true);
        goToStep(curTree, getTS(curTree).stepIdx + 1);
      } else if (e.key === "ArrowLeft" || e.key === "ArrowUp") {
        e.preventDefault();
        setHintDismissed(true);
        goToStep(curTree, getTS(curTree).stepIdx - 1);
      }
    };
    window.addEventListener("keydown", h);
    return () => window.removeEventListener("keydown", h);
  });

  // ── Wheel / zoom ───────────────────────────────────────────────────────────
  useEffect(() => {
    const el = canvasRef.current;
    if (!el) return;
    const h = (e) => {
      e.preventDefault();
      const rect = el.getBoundingClientRect();
      const mx = e.clientX - rect.left, my = e.clientY - rect.top;
      let delta = -e.deltaY;
      if (e.ctrlKey) { delta = delta * 0.008; }
      else { const px = e.deltaMode === 1 ? delta * 16 : delta; delta = px * 0.001; }
      delta = Math.max(-0.15, Math.min(0.15, delta));
      // Read current values synchronously from refs — avoids nested setState side effect.
      const z  = zoomLive.current;
      const p  = panLive.current;
      const nz = Math.max(0.08, Math.min(4, z * (1 + delta)));
      const sc = nz / z;
      // Point under cursor in SVG space stays fixed: newPan = cursor - sc*(cursor - pan)
      setZoom(nz);
      setPan({ x: mx - sc * (mx - p.x), y: my - sc * (my - p.y) });
    };
    el.addEventListener("wheel", h, { passive: false });
    return () => el.removeEventListener("wheel", h);
  }, []);

  // ── Pan handlers ───────────────────────────────────────────────────────────
  const onMouseDown = (e) => {
    isDragging.current = true;
    setCursorGrabbing(true);
    dragStart.current = { x: e.clientX, y: e.clientY, px: pan.x, py: pan.y };
  };
  const onMouseMove = (e) => {
    if (!isDragging.current) return;
    setPan({ x: dragStart.current.px + e.clientX - dragStart.current.x, y: dragStart.current.py + e.clientY - dragStart.current.y });
  };
  const onMouseUp = () => { isDragging.current = false; setCursorGrabbing(false); };

  // ── Tree centering ─────────────────────────────────────────────────────────
  // Centers the root node horizontally in the visible canvas area.
  // Root is always at SVG x = treeWidth/2, so pan.x = (canvasW - treeWidth) / 2.
  const centerTree = useCallback((treeIdx) => {
    const el = canvasRef.current;
    if (!el || !trees[treeIdx]) return;
    const canvasW = el.getBoundingClientRect().width;
    const tw = computeTreeWidth(trees[treeIdx]);
    setPan({ x: (canvasW - tw) / 2, y: 20 });
  }, [trees]);

  useLayoutEffect(() => {
    centerTree(curTree);
  }, [curTree, trees, centerTree]);

  // ── Drag-to-complete global mouseup ───────────────────────────────────────
  useEffect(() => {
    const onMouseUp = () => {
      if (!dragRef.current.active) return;
      const { startIdx, endIdx, moved } = dragRef.current;
      dragRef.current = { active: false, startIdx: null, endIdx: null, moved: false };
      setDragRange(null);
      if (moved && startIdx !== null && endIdx !== null) {
        setHintDismissed(true);
        const lo = Math.min(startIdx, endIdx);
        const hi = Math.max(startIdx, endIdx);
        for (let j = lo; j <= hi; j++) {
          instantComplete(j);
        }
      }
    };
    window.addEventListener("mouseup", onMouseUp);
    return () => window.removeEventListener("mouseup", onMouseUp);
  }, [instantComplete]);

  // ── Auto-advance to step 0 when first tree arrives ────────────────────────
  // Shows the root node with '?' immediately so the canvas is never empty.
  useEffect(() => {
    if (trees[0] && !treeStates[0]) {
      goToStep(0, 0);
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [trees[0]]);

  // ── Scroll-down hint — appears when all trees done, fades after 5s or scroll ─
  useEffect(() => {
    const allDone = trees.length > 0 && !growing && !buildProgress &&
      trees.every((t, i) => {
        if (!t) return false;
        const s = treeStates[i];
        const steps = getSteps(i);
        return s && s.stepIdx >= steps.length - 1 && steps.length > 0;
      });
    if (!allDone) { setScrollHint(false); return; }
    setScrollHint(true);
    const timer = setTimeout(() => setScrollHint(false), 5000);
    const el = predSectionRef.current;
    if (!el) return () => clearTimeout(timer);
    const obs = new IntersectionObserver(([entry]) => {
      if (entry.isIntersecting) { setScrollHint(false); clearTimeout(timer); }
    }, { threshold: 0.1 });
    obs.observe(el);
    return () => { clearTimeout(timer); obs.disconnect(); };
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [trees, treeStates, growing, buildProgress]);

  // ── File drop handling ─────────────────────────────────────────────────────
  const openFile = useCallback((file) => {
    if (!file || !file.name.toLowerCase().endsWith(".csv")) return;
    const reader = new FileReader();
    reader.onload = (e) => {
      const { data: rawRows, meta } = Papa.parse(e.target.result, { header: true, skipEmptyLines: true });
      if (!rawRows.length) return;
      const headers    = meta.fields;
      const naStats    = detectNAs(rawRows, headers);
      const initTarget = headers[headers.length - 1];
      setCsvModal({
        fileName: file.name, rawRows, headers, naStats,
        selectedTarget: initTarget,
        naStrategy: "drop", sampleMode: Math.min(1000, rawRows.length),
        selectedColumns: headers,
        taskType: detectTaskType(rawRows, initTarget),
      });
    };
    reader.readAsText(file);
  }, []);

  const handleDrop = useCallback((e) => {
    e.preventDefault();
    setDragOver(false);
    const file = e.dataTransfer.files[0];
    openFile(file);
  }, [openFile]);

  // ── CSV modal confirm ──────────────────────────────────────────────────────
  const handleDataConfirm = useCallback((data, features, targetCol, name, totalRows, sampledRows, classLabels, taskType = "classification", originalTarget = null) => {
    setCustomDataset({ data, features, targetCol, name, totalRows, sampledRows, classLabels, taskType, originalTarget });
    setSelectedSampleIdx(null);
    setCsvModal(null);
    buildForestWithData(data, features, targetCol, taskType);
  }, [buildForestWithData]);

  // ── Calculations sidebar resize/toggle ────────────────────────────────────
  // Single handler: drag = resize, click (no drag) = collapse toggle.
  const onCalcHandleMouseDown = useCallback((e) => {
    if (e.button !== 0) return;
    e.preventDefault();
    const startX   = e.clientX;
    const startW   = calcPanelWidth;
    const expanded = calcExpanded;
    let moved = false;

    const onMove = (ev) => {
      if (!expanded) return; // don't resize when collapsed
      if (!moved && Math.abs(ev.clientX - startX) > 3) moved = true;
      if (!moved) return;
      const delta  = startX - ev.clientX; // left-edge drag: left = wider
      setCalcPanelWidth(Math.max(180, Math.min(400, startW + delta)));
    };
    const onUp = () => {
      document.removeEventListener("mousemove", onMove);
      document.removeEventListener("mouseup",   onUp);
      if (!moved) setCalcExpanded(x => !x); // plain click = toggle
    };
    document.addEventListener("mousemove", onMove);
    document.addEventListener("mouseup",   onUp);
  }, [calcExpanded, calcPanelWidth]);

  // ── Derived render values ──────────────────────────────────────────────────
  const currentTree = trees[curTree];
  const treeWidth   = currentTree
    ? computeTreeWidth(currentTree)
    : X_PAD * 2 + Math.pow(2, maxDepth) * LEAF_SPACING;
  const svgH        = (maxDepth + 1) * Y_GAP + 60;
  const allNodes    = currentTree ? flattenNodes(currentTree) : [];
  const positions   = currentTree ? computePositions(currentTree) : {};
  const ts          = getTS(curTree);
  const visibleSet  = new Set(ts.visibleIds);
  const currentNode = allNodes.find(n => n.id === ts.nodeId);
  const totalSteps  = getSteps(curTree).length;

  const completedTrees = trees.map((t, i) => {
    if (!t) return null;
    const s = treeStates[i];
    const steps = getSteps(i);
    const done  = s && s.stepIdx >= steps.length - 1 && steps.length > 0;
    return done ? { idx: i, prediction: getTreePrediction(t) } : null;
  }).filter(Boolean);

  const hasEnsemble = completedTrees.length >= 1;

  // ── Sample-based prediction derived values ─────────────────────────────────
  // Clamp index in case data shrank after a CSV reload
  const safeSampleIdx = selectedSampleIdx !== null && selectedSampleIdx < activeData.length
    ? selectedSampleIdx : null;
  const selectedSample = safeSampleIdx !== null ? activeData[safeSampleIdx] : null;

  // Per-tree predictions for the selected sample (only on completed trees)
  const sampleTreePreds = useMemo(() => {
    if (!selectedSample) return [];
    return completedTrees.map(ct => ({
      ...ct,
      samplePred: predictRow(trees[ct.idx], selectedSample, activeFeatures),
    }));
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedSample, completedTrees.length, trees]);

  const sampleVotesPerClass = {};
  if (activeTaskType === "classification") {
    sampleTreePreds.forEach(t => {
      sampleVotesPerClass[t.samplePred] = (sampleVotesPerClass[t.samplePred] ?? 0) + 1;
    });
  }
  const sampleMajority  = activeTaskType === "classification"
    ? Object.entries(sampleVotesPerClass).sort((a, b) => b[1] - a[1])[0]?.[0]
    : null;
  const sampleMean      = activeTaskType === "regression" && sampleTreePreds.length > 0
    ? sampleTreePreds.reduce((s, t) => s + t.samplePred, 0) / sampleTreePreds.length
    : null;
  const sampleTrueLabel = selectedSample ? selectedSample[activeTargetCol] : null;
  const sampleCorrect   = sampleMajority !== undefined && sampleMajority === sampleTrueLabel;

  // Path of node IDs the sample follows through the currently visible tree
  const samplePath = useMemo(() => {
    if (!selectedSample || !currentTree) return new Set();
    return new Set(getSamplePath(currentTree, selectedSample, activeFeatures));
  }, [selectedSample, currentTree, activeFeatures]);

  // Classification: forest accuracy on training data
  const forestAccuracy = useMemo(() => {
    if (activeTaskType !== "classification") return null;
    if (completedTrees.length === 0 || activeData.length === 0) return null;
    let correct = 0;
    for (const row of activeData) {
      const votes = {};
      completedTrees.forEach(ct => {
        const pred = predictRow(trees[ct.idx], row, activeFeatures);
        votes[pred] = (votes[pred] ?? 0) + 1;
      });
      const pred = Object.entries(votes).sort((a, b) => b[1] - a[1])[0]?.[0];
      if (pred === row[activeTargetCol]) correct++;
    }
    return (correct / activeData.length * 100).toFixed(1);
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [activeTaskType, completedTrees.length, trees, activeData, activeFeatures, activeTargetCol]);

  // Regression: R² and RMSE on training data
  const forestRegMetrics = useMemo(() => {
    if (activeTaskType !== "regression") return null;
    if (completedTrees.length === 0 || activeData.length === 0) return null;
    const preds = activeData.map(row => {
      const treePreds = completedTrees.map(ct => predictRow(trees[ct.idx], row, activeFeatures));
      return treePreds.reduce((s, v) => s + v, 0) / treePreds.length;
    });
    const actual  = activeData.map(r => r[activeTargetCol]);
    const n       = actual.length;
    const meanA   = actual.reduce((s, v) => s + v, 0) / n;
    const ssTot   = actual.reduce((s, v) => s + (v - meanA) ** 2, 0);
    const ssRes   = actual.reduce((s, v, i) => s + (v - preds[i]) ** 2, 0);
    const r2      = ssTot > 0 ? 1 - ssRes / ssTot : 1;
    const rmse    = Math.sqrt(ssRes / n);
    return { r2: r2.toFixed(3), rmse: formatRegVal(rmse) };
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [activeTaskType, completedTrees.length, trees, activeData, activeFeatures, activeTargetCol]);

  const filledBInfo    = bootstrapInfo.filter(Boolean);
  const avgOobAccuracy = activeTaskType === "classification" && filledBInfo.length > 0 && filledBInfo.some(b => b.oobAccuracy > 0)
    ? (filledBInfo.reduce((s, b) => s + (b.oobAccuracy ?? 0), 0) / filledBInfo.length * 100).toFixed(1)
    : null;
  const validOobReg    = filledBInfo.filter(b => b.oobR2 != null);
  const avgOobR2       = activeTaskType === "regression" && validOobReg.length > 0
    ? (validOobReg.reduce((s, b) => s + b.oobR2, 0) / validOobReg.length).toFixed(3)
    : null;
  const avgOobRMSE     = activeTaskType === "regression" && validOobReg.length > 0
    ? formatRegVal(validOobReg.reduce((s, b) => s + b.oobRMSE, 0) / validOobReg.length)
    : null;

  const curBootstrap  = bootstrapInfo[curTree] ?? null;
  const TOTAL_SAMPLES = activeData.length;

  const inp = {
    padding: "6px 10px", borderRadius: 8, background: "#161c2a",
    border: "none", color: C.text, fontSize: 12,
    fontFamily: "'JetBrains Mono',monospace", outline: "none",
    boxShadow: "inset 0 0 0 1px rgba(255,255,255,0.06)",
  };

  // ── Render ─────────────────────────────────────────────────────────────────
  return (
    <div
      style={{ minHeight: "100vh", background: C.bg, color: C.text }}
      onDragOver={e => { e.preventDefault(); setDragOver(true); }}
      onDragLeave={() => setDragOver(false)}
      onDrop={handleDrop}
    >
      {/* Build progress overlay */}
      {buildProgress && (
        <div style={{
          position: "fixed", inset: 0, zIndex: 200,
          background: "rgba(10,14,23,0.82)",
          backdropFilter: "blur(5px)",
          display: "flex", flexDirection: "column",
          alignItems: "center", justifyContent: "center",
          gap: 18,
          pointerEvents: "none",
        }}>
          <div style={{
            fontSize: 15, fontWeight: 700, color: C.text,
          }}>
            Building forest… {buildProgress.done}/{buildProgress.total} trees
          </div>
          {/* Progress bar */}
          <div style={{
            width: 320, height: 6, background: C.border,
            borderRadius: 3, overflow: "hidden",
          }}>
            <div style={{
              height: "100%",
              width: `${buildProgress.total > 0 ? (buildProgress.done / buildProgress.total) * 100 : 0}%`,
              background: `linear-gradient(90deg, ${C.accent}, ${C.green})`,
              borderRadius: 3,
              transition: "width 0.15s ease-out",
            }} />
          </div>
          <div style={{ fontSize: 10, color: C.dim }}>
            {buildProgress.total - buildProgress.done > 0
              ? `${buildProgress.total - buildProgress.done} tree${buildProgress.total - buildProgress.done === 1 ? "" : "s"} remaining`
              : "Finalising…"}
          </div>
        </div>
      )}

      {/* Drag overlay */}
      {dragOver && (
        <div style={{
          position: "fixed", inset: 0, zIndex: 50, background: `${C.accent}18`,
          border: `2px dashed ${C.accent}`, borderRadius: 8,
          display: "flex", alignItems: "center", justifyContent: "center",
          pointerEvents: "none",
        }}>
          <span style={{ fontSize: 18, color: C.accent, fontWeight: 700 }}>Drop CSV to load dataset</span>
        </div>
      )}

      {/* CSV modal */}
      {csvModal && (
        <DataModal
          modal={csvModal}
          onUpdate={patch => setCsvModal(prev => ({ ...prev, ...patch }))}
          onConfirm={handleDataConfirm}
          onCancel={() => setCsvModal(null)}
        />
      )}

      {/* Shared hidden CSV input — triggered by both the dropdown and the header button */}
      <input ref={fileInputRef} type="file" accept=".csv" style={{ display: "none" }}
        onChange={e => { openFile(e.target.files[0]); e.target.value = ""; }} />

      {/* Global Header */}
      <GlobalHeader
        right={
          <button
            onClick={() => fileInputRef.current?.click()}
            title="Upload a CSV file"
            onMouseEnter={e => { e.currentTarget.style.color = C.text; }}
            onMouseLeave={e => { e.currentTarget.style.color = C.dim; }}
            style={{
              background: "none", border: "none", cursor: "pointer",
              fontSize: 12, color: C.dim, fontWeight: 500,
              padding: 0, transition: "color 0.15s", fontFamily: "inherit",
            }}
          >
            ↑ Upload CSV
          </button>
        }
        infoBar={(() => {
          const algoLabel   = mode === "decision-tree" ? "Decision Tree" : mode === "bagging" ? "Bagging" : "Random Forest";
          const targetDisp  = customDataset?.originalTarget
            ?? (builtinDataset === "music" ? "genre" : builtinDataset === "salary" ? "Salary" : "Disease");
          const nClasses    = Array.isArray(classLabels) ? classLabels.length : Object.keys(classLabels ?? {}).length;
          const taskTag     = activeTaskType === "regression"
            ? "Regression"
            : nClasses === 2 ? "Binary" : `${nClasses}-class`;
          const sampleCount = activeData.length;
          const disabled    = !!buildProgress || growing;

          const Badge = ({ children }) => (
            <span style={{
              display: "inline-flex", alignItems: "center",
              background: "rgba(255,255,255,0.07)",
              borderRadius: 6, padding: "4px 10px",
              fontSize: 11, whiteSpace: "nowrap",
            }}>
              {children}
            </span>
          );

          return (
            <>
              {/* Left: algorithm name + description tooltip */}
              <div style={{ display: "flex", alignItems: "center", gap: 0 }}>
                <span style={{
                  fontSize: 18, fontWeight: 700, color: C.text,
                  letterSpacing: "-0.3px", lineHeight: 1,
                }}>
                  {algoLabel}
                </span>
                <AlgoTooltip mode={mode} />
              </div>

              {/* Right: dataset pill + stat badges */}
              <div style={{ display: "flex", alignItems: "center", gap: 8, flexShrink: 0 }}>
                {/* Dataset label + dropdown pill, inline */}
                <div style={{ display: "flex", alignItems: "center", gap: 7 }}>
                  <span style={{ fontSize: 9, color: C.muted, letterSpacing: "0.04em", whiteSpace: "nowrap" }}>
                    Dataset
                  </span>
                  <div ref={rfDropRef} style={{ position: "relative", display: "inline-flex", alignItems: "center" }}>
                    {/* Trigger button */}
                    <button
                      disabled={disabled}
                      onClick={e => {
                        if (disabled) return;
                        const r = e.currentTarget.getBoundingClientRect();
                        setRfDropPos({ top: r.bottom + 4, left: r.left });
                        setRfDropOpen(v => !v);
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
                        : builtinDataset === "music"
                          ? "Built-in: Music Genres (Multiclass classification)"
                          : builtinDataset === "salary"
                            ? "Built-in: Salary (Regression)"
                            : "Built-in: Heart Disease (Binary classification)"}
                    </button>
                    <span style={{ position: "absolute", right: 9, top: "50%", transform: "translateY(-50%)",
                      pointerEvents: "none", color: C.dimmer, fontSize: 10 }}>▾</span>
                  </div>
                </div>

                {/* Stat badges */}
                <Badge>
                  <span style={{ color: C.dim, marginRight: 4 }}>Target</span>
                  <span style={{ color: C.text, fontWeight: 600 }}>{targetDisp}</span>
                </Badge>
                <Badge><span style={{ color: C.text, fontWeight: 600 }}>{taskTag}</span></Badge>
                <Badge>
                  <span style={{ color: C.text, fontWeight: 600 }}>{sampleCount.toLocaleString()}</span>
                  <span style={{ color: C.dim, marginLeft: 4 }}>samples</span>
                </Badge>
              </div>
            </>
          );
        })()}
      />

      {/* Controls toolbar */}
      <div style={{
        display: "flex", alignItems: "flex-end", gap: 10,
        padding: "12px 20px",
        background: C.panel,
        boxShadow: "0 1px 0 rgba(255,255,255,0.03), 0 4px 20px rgba(0,0,0,0.35)",
        flexWrap: "wrap",
        position: "relative", zIndex: 20,
      }}>
        {/* Model hyperparameters */}
        <div data-tutorial="hyperparams" style={{ display: "flex", alignItems: "flex-end", gap: 12, flex: 1, flexWrap: "wrap" }}>
          {/* Max depth — always unlocked */}
          <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
            <label style={{ fontSize: 9, color: C.muted, fontWeight: 400 }}>Max depth</label>
            <input type="number" min={1} max={activeFeatures.length} value={maxDepthStr} disabled={growing}
              onChange={e => setMaxDepthStr(e.target.value)}
              onKeyDown={e => e.stopPropagation()}
              onBlur={() => {
                const v = Math.max(1, Math.min(activeFeatures.length, parseInt(maxDepthStr, 10) || 1));
                setMaxDepth(v); setMaxDepthStr(String(v));
              }}
              style={{ ...inp, width: 58 }} />
          </div>

          {/* Max features — locked on decision-tree and bagging */}
          {lockedMaxFeatures ? (
            // Outer wrapper: handles hover + positioning. No opacity here so tooltip is unaffected.
            <div
              style={{ position: "relative", cursor: "default" }}
              onMouseEnter={e => setLockedParamTooltip({ which: "maxFeatures", ...tooltipPosition(e.currentTarget.getBoundingClientRect(), { prefer: 'above', width: 240, height: 58 }) })}
              onMouseLeave={() => setLockedParamTooltip(null)}
            >
              {/* Dimmed control content */}
              <div style={{ display: "flex", flexDirection: "column", gap: 4, opacity: 0.45, pointerEvents: "none", userSelect: "none" }}>
                <label style={{ fontSize: 9, color: C.muted, fontWeight: 400, display: "flex", alignItems: "center", gap: 4 }}>
                  Max features <span style={{ fontSize: 9, lineHeight: 1 }}>🔒</span>
                </label>
                <select value="all" disabled style={{ ...inp, cursor: "not-allowed", width: "auto" }}>
                  <option value="all">p (all) → {activeFeatures.length}</option>
                </select>
              </div>
            </div>
          ) : (
            <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
              <label style={{ fontSize: 9, color: C.muted, fontWeight: 400 }}>Max features</label>
              <select value={featureSubset} disabled={growing} onChange={e => setFeatureSubset(e.target.value)} style={{ ...inp, cursor: "pointer" }}>
                {Object.entries(FEATURE_SUBSET_OPTIONS).map(([k, v]) => (
                  <option key={k} value={k}>{v.label} → {v.fn(activeFeatures.length)}</option>
                ))}
              </select>
            </div>
          )}

          {/* Trees — locked on decision-tree */}
          {lockedNEstimators ? (
            // Outer wrapper: handles hover + positioning. No opacity here so tooltip is unaffected.
            <div
              style={{ position: "relative", cursor: "default" }}
              onMouseEnter={e => setLockedParamTooltip({ which: "trees", ...tooltipPosition(e.currentTarget.getBoundingClientRect(), { prefer: 'above', width: 240, height: 40 }) })}
              onMouseLeave={() => setLockedParamTooltip(null)}
            >
              {/* Dimmed control content */}
              <div style={{ display: "flex", flexDirection: "column", gap: 4, opacity: 0.45, pointerEvents: "none", userSelect: "none" }}>
                <label style={{ fontSize: 9, color: C.muted, fontWeight: 400, display: "flex", alignItems: "center", gap: 4 }}>
                  Trees <span style={{ fontSize: 9, lineHeight: 1 }}>🔒</span>
                </label>
                <input type="number" value={1} disabled style={{ ...inp, width: 58, cursor: "not-allowed" }} />
              </div>
            </div>
          ) : (
            <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
              <label style={{ fontSize: 9, color: C.muted, fontWeight: 400 }}>Trees</label>
              <input type="number" min={1} max={100} value={nEstimatorsStr} disabled={growing}
                onChange={e => setNEstimatorsStr(e.target.value)}
                onKeyDown={e => e.stopPropagation()}
                onBlur={() => {
                  const v = Math.max(1, Math.min(100, parseInt(nEstimatorsStr, 10) || 1));
                  setNEstimators(v); setNEstimatorsStr(String(v));
                }}
                style={{ ...inp, width: 58 }} />
            </div>
          )}

          {/* Speed — always unlocked */}
          <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
            <label style={{ fontSize: 9, color: C.muted, fontWeight: 400 }}>Speed</label>
            <select value={speed} onChange={e => setSpeed(+e.target.value)} style={{ ...inp, cursor: "pointer", width: 64 }}>
              {[0.5, 1, 2, 4].map(s => <option key={s} value={s}>{s}×</option>)}
            </select>
          </div>
        </div>

        {/* Buttons */}
        <div style={{ display: "flex", gap: 8, alignItems: "flex-end" }}>
          <button disabled={!!buildProgress} onClick={buildForest}
            style={{ padding: "7px 14px", borderRadius: 8, border: `1px solid ${C.border}`,
              background: "transparent", color: C.dim, fontSize: 11, fontFamily: "inherit",
              cursor: buildProgress ? "default" : "pointer", opacity: buildProgress ? 0.4 : 1,
              transition: "color 0.15s, border-color 0.15s" }}
            onMouseEnter={e => { if (!buildProgress) { e.currentTarget.style.color = C.text; e.currentTarget.style.borderColor = C.dim; } }}
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
            <button data-tutorial="grow-button" disabled={!!buildProgress} onClick={() => { setHintDismissed(true); autoGrowAll(); }}
              title={tutorial.active && (tutorial.stepIdx === 2 || tutorial.stepIdx === 4) ? "Use → during the tutorial" : undefined}
              style={{ padding: "7px 18px", borderRadius: 8, border: "none",
                background: buildProgress ? C.border : `linear-gradient(135deg,${C.accent},#d97706)`,
                color: buildProgress ? C.dim : "#000",
                fontSize: 11, fontFamily: "inherit", cursor: buildProgress ? "default" : "pointer",
                fontWeight: 600, transition: "box-shadow 0.15s, filter 0.15s, opacity 0.15s",
                opacity: (tutorial.active && (tutorial.stepIdx === 2 || tutorial.stepIdx === 4)) ? 0.4 : 1,
                pointerEvents: (tutorial.active && (tutorial.stepIdx === 2 || tutorial.stepIdx === 4)) ? "none" : undefined,
              }}
              onMouseEnter={e => { if (!buildProgress) { e.currentTarget.style.boxShadow = "0 0 12px rgba(245,158,11,0.3)"; e.currentTarget.style.filter = "brightness(1.1)"; } }}
              onMouseLeave={e => { e.currentTarget.style.boxShadow = "none"; e.currentTarget.style.filter = "none"; }}>
              Grow
            </button>
          )}

          <button data-tutorial="complete-all" disabled={!trees.length || growing || !!buildProgress} onClick={growAllInstant}
            style={{ padding: "7px 18px", borderRadius: 8, border: "none",
              background: !trees.length || growing || buildProgress ? C.border : `linear-gradient(135deg,${C.accent},#d97706)`,
              color: !trees.length || growing || buildProgress ? C.dim : "#000",
              fontSize: 11, fontFamily: "inherit",
              cursor: !trees.length || growing || buildProgress ? "default" : "pointer",
              fontWeight: 600, transition: "box-shadow 0.15s, filter 0.15s" }}
            onMouseEnter={e => { if (!trees.length || growing || buildProgress) return; e.currentTarget.style.boxShadow = "0 0 12px rgba(245,158,11,0.3)"; e.currentTarget.style.filter = "brightness(1.1)"; }}
            onMouseLeave={e => { e.currentTarget.style.boxShadow = "none"; e.currentTarget.style.filter = "none"; }}>
            Complete All
          </button>
        </div>
      </div>

      {/* Unified navigation bar */}
      <div style={{
        display: "flex", alignItems: "center",
        padding: "0 16px",
        boxShadow: "0 1px 0 rgba(255,255,255,0.03), 0 2px 12px rgba(0,0,0,0.25)",
        background: C.bg, minHeight: 40, overflow: "hidden",
        position: "relative", zIndex: 4,
      }}>
        {/* Scrollable tabs area — only this region scrolls */}
        {nEstimators > 1 && (() => {
          // Compute selection overlay rect from tab DOM refs
          let selRect = null;
          if (dragRange !== null && tabScrollRef.current) {
            const containerRect = tabScrollRef.current.getBoundingClientRect();
            const lo = Math.min(dragRange.start, dragRange.end);
            const hi = Math.max(dragRange.start, dragRange.end);
            const startEl = tabRefs.current[lo];
            const endEl   = tabRefs.current[hi];
            if (startEl && endEl) {
              const sR = startEl.getBoundingClientRect();
              const eR = endEl.getBoundingClientRect();
              selRect = {
                left:   sR.left  - containerRect.left - 4,
                top:    sR.top   - containerRect.top  - 4,
                width:  eR.right - sR.left + 8,
                height: sR.height + 8,
              };
            }
          }
          return (
            <div style={{ position: "relative", flex: 1, minWidth: 0 }}>
              {selRect && (
                <div style={{
                  position: "absolute",
                  left: selRect.left, top: selRect.top,
                  width: selRect.width, height: selRect.height,
                  border: `1.5px dashed ${C.accent}`,
                  borderRadius: 10, pointerEvents: "none", zIndex: 10,
                  boxShadow: `0 0 8px ${C.accent}33`,
                }} />
              )}
              <div data-tutorial="tree-tabs" ref={tabScrollRef} style={{
                display: "flex", alignItems: "center", gap: 3,
                overflowX: "auto", padding: "6px 0",
                scrollbarWidth: "none", msOverflowStyle: "none",
                cursor: dragRange !== null ? "grabbing" : "default",
              }}>
                {trees.map((t, i) => {
                  const s       = treeStates[i];
                  const done    = t && s && s.stepIdx >= getSteps(i).length - 1 && getSteps(i).length > 0;
                  const loading = !t && buildProgress !== null;
                  const inSel   = dragRange !== null && i >= Math.min(dragRange.start, dragRange.end) && i <= Math.max(dragRange.start, dragRange.end);
                  return (
                    <button key={i}
                      ref={el => { tabRefs.current[i] = el; }}
                      onClick={() => {
                        if (!growing && !buildProgress && t && !dragRef.current.moved) setCurTree(i);
                      }}
                      onDoubleClick={() => {
                        if (!growing && !buildProgress && t) { setHintDismissed(true); setCurTree(i); instantComplete(i); }
                      }}
                      onMouseDown={e => {
                        if (loading || !t || growing || buildProgress) return;
                        e.preventDefault();
                        dragRef.current = { active: true, startIdx: i, endIdx: i, moved: false };
                        setDragRange({ start: i, end: i });
                        setTabTooltip(null);
                      }}
                      onMouseEnter={e => {
                        if (dragRef.current.active && t && !loading && !growing && !buildProgress) {
                          dragRef.current.endIdx = i;
                          dragRef.current.moved  = i !== dragRef.current.startIdx;
                          setDragRange({ start: dragRef.current.startIdx, end: i });
                        } else if (t && !loading) {
                          const r = e.currentTarget.getBoundingClientRect();
                          setTabTooltip(tooltipPosition(r, { prefer: 'below', gap: 6, width: 240, height: 28 }));
                        }
                      }}
                      onMouseLeave={() => {
                        if (!dragRef.current.active) setTabTooltip(null);
                      }}
                      style={{
                        padding: "3px 10px", borderRadius: 8, flexShrink: 0,
                        border: "1px solid transparent",
                        background: i === curTree ? C.accent
                          : inSel ? `${C.accent}18`
                          : done  ? `${C.accent}22`
                          : "rgba(255,255,255,0.04)",
                        color: i === curTree ? "#000" : done ? C.accent : loading ? C.dimmer : C.dim,
                        fontSize: 10, fontFamily: "inherit",
                        cursor: (loading || !t) ? "default" : "pointer",
                        fontWeight: i === curTree ? 700 : 400,
                        transition: "background 0.15s ease-out, color 0.15s ease-out",
                        opacity: loading ? 0.5 : 1,
                        userSelect: "none",
                      }}>
                      T{i + 1}{done ? " ✓" : ""}
                    </button>
                  );
                })}
              </div>
            </div>
          );
        })()}
        {nEstimators === 1 && <div style={{ flex: 1 }} />}

        {/* Pinned right side — never scrolls: Complete all | step controls | bootstrap | Reset */}
        <div style={{
          display: "flex", alignItems: "center", gap: 8,
          flexShrink: 0, paddingLeft: 10, paddingTop: 6, paddingBottom: 6,
          borderLeft: nEstimators > 1 ? `1px solid rgba(255,255,255,0.06)` : "none",
          marginLeft: nEstimators > 1 ? 6 : 0,
        }}>
          {/* Step controls */}
          <button disabled={growing || !!buildProgress || ts.stepIdx <= -1} onClick={() => { setHintDismissed(true); goToStep(curTree, ts.stepIdx - 1); }}
            style={{ ...inp, cursor: (growing || buildProgress || ts.stepIdx <= -1) ? "default" : "pointer", padding: "3px 10px", opacity: (growing || buildProgress || ts.stepIdx <= -1) ? 0.3 : 1, fontSize: 11, transition: "background 0.15s" }}
            onMouseEnter={e => { if (!growing && !buildProgress && ts.stepIdx > -1) e.currentTarget.style.background = "#1e2840"; }}
            onMouseLeave={e => { e.currentTarget.style.background = "#161c2a"; }}>◀</button>
          <span style={{ fontSize: 10, color: C.muted, minWidth: 72, textAlign: "center", userSelect: "none" }}>
            {ts.stepIdx === -1 ? "ready" : `${ts.stepIdx + 1} / ${totalSteps}`}
          </span>
          <button disabled={growing || !!buildProgress || ts.stepIdx >= totalSteps - 1} onClick={() => { setHintDismissed(true); goToStep(curTree, ts.stepIdx + 1); }}
            style={{ ...inp, cursor: (growing || buildProgress || ts.stepIdx >= totalSteps - 1) ? "default" : "pointer", padding: "3px 10px", opacity: (growing || buildProgress || ts.stepIdx >= totalSteps - 1) ? 0.3 : 1, fontSize: 11, transition: "background 0.15s" }}
            onMouseEnter={e => { if (!growing && !buildProgress && ts.stepIdx < totalSteps - 1) e.currentTarget.style.background = "#1e2840"; }}
            onMouseLeave={e => { e.currentTarget.style.background = "#161c2a"; }}>▶</button>

          {/* Bootstrap info */}
          {curBootstrap && (
            <span style={{ fontSize: 9, color: C.muted, whiteSpace: "nowrap" }}>
              {curBootstrap.inBag}/{TOTAL_SAMPLES} in-bag · {curBootstrap.oob} OOB
              {curBootstrap.oobAccuracy > 0 ? ` · acc=${curBootstrap.oobAccuracy}` : ""}
            </span>
          )}

          {/* Reset — single click: current tree · double click: all trees */}
          <button
            disabled={growing}
            onMouseEnter={e => {
              setResetTooltip(tooltipPosition(e.currentTarget.getBoundingClientRect(), { prefer: 'above', width: 272, height: 28 }));
              if (!growing) e.currentTarget.style.background = "#1e2840";
            }}
            onMouseLeave={e => {
              setResetTooltip(null);
              e.currentTarget.style.background = "#161c2a";
            }}
            onClick={() => {
              if (resetClickRef.current) return;
              resetClickRef.current = setTimeout(() => {
                resetClickRef.current = null;
                setTS(curTree, EMPTY_TS);
              }, 220);
            }}
            onDoubleClick={() => {
              clearTimeout(resetClickRef.current);
              resetClickRef.current = null;
              pausedAtRef.current = null;
              setTreeStates({});
              setCurTree(0);
            }}
            style={{ ...inp, cursor: growing ? "default" : "pointer", padding: "3px 10px", fontSize: 10, opacity: growing ? 0.3 : 1, transition: "background 0.15s" }}>
            Reset
          </button>
        </div>
      </div>

      {/* ── Training section header ───────────────────────────────────────────── */}
      <div style={{
        padding: "10px 20px 9px",
        borderTop: "1px solid rgba(255,255,255,0.05)",
        display: "flex",
        alignItems: "center",
        gap: 12,
      }}>
        <span style={{ fontSize: 11, fontWeight: 700, color: C.accent, letterSpacing: "0.06em", flexShrink: 0 }}>
          Step 1
        </span>
        <span style={{ fontSize: 11, fontWeight: 700, color: C.text, letterSpacing: "0.04em", flexShrink: 0 }}>
          Training
        </span>
        <div style={{ flex: 1, height: 1, background: `linear-gradient(to right, ${C.accent}40, transparent)` }} />
      </div>

      {/* Canvas row — tree canvas + calculations sidebar side by side */}
      <div style={{ display: "flex", alignItems: "stretch", height: Math.min(500, svgH + 40) }}>

      {/* SVG canvas */}
      <div ref={canvasRef} data-tutorial="tree-canvas"
        onMouseDown={onMouseDown} onMouseMove={onMouseMove}
        onMouseUp={onMouseUp} onMouseLeave={onMouseUp}
        style={{
          flex: 1, minWidth: 0,
          overflow: "hidden", cursor: cursorGrabbing ? "grabbing" : "grab",
          background: "#080c14",
          boxShadow: "0 2px 16px rgba(0,0,0,0.4)",
          position: "relative",
        }}>

        {/* Welcome overlay — shown on step 0, dismissed on first action or when any tree completes.
            Also shown during tutorial steps 3 and 5 (indices 2 and 4) so the root "?" node and
            this message stay visible as the coachmark layers on top. */}
        {!hintDismissed &&
          (!tutorial.active || tutorial.stepIdx === 2 || tutorial.stepIdx === 4) &&
          completedTrees.length === 0 && !growing && !buildProgress && trees[0] && (() => {
          const growDisabledByTutorial = tutorial.active && (tutorial.stepIdx === 2 || tutorial.stepIdx === 4);
          return (
            <div style={{
              position: "absolute", inset: 0, display: "flex", flexDirection: "column",
              alignItems: "center", justifyContent: "center", pointerEvents: "none",
              zIndex: 5,
            }}>
              <div style={{
                background: "rgba(10,14,23,0.82)",
                border: `1px solid rgba(255,255,255,0.08)`,
                borderRadius: 16,
                padding: "22px 32px",
                maxWidth: 420, textAlign: "center",
                backdropFilter: "blur(6px)",
                boxShadow: "0 8px 40px rgba(0,0,0,0.6)",
              }}>
                <div style={{ fontSize: 13, color: C.text, fontWeight: 600, marginBottom: growDisabledByTutorial ? 0 : 10, lineHeight: 1.5 }}>
                  Use <span style={{ color: C.accent, fontWeight: 700 }}>→</span> to step through the tree
                  {!growDisabledByTutorial && (
                    <>, or press <span style={{ color: C.accent, fontWeight: 700 }}>▶ Grow</span> to watch it build automatically.</>
                  )}
                  {growDisabledByTutorial && "."}
                </div>
                {!growDisabledByTutorial && (
                  <div style={{ fontSize: 10.5, color: C.dim, lineHeight: 1.7 }}>
                    You can also change the dataset above to visualize on your own data.
                  </div>
                )}
              </div>
            </div>
          );
        })()}

        <svg width={treeWidth} height={svgH}
          style={{ transform: `translate(${pan.x}px,${pan.y}px) scale(${zoom})`, transformOrigin: "0 0", display: "block" }}>
          <defs>
            <filter id="ng" x="-50%" y="-50%" width="200%" height="200%">
              <feDropShadow dx="0" dy="0" stdDeviation="6" floodColor={C.accent} floodOpacity="0.22" />
            </filter>
            <filter id="ng-p" x="-50%" y="-50%" width="200%" height="200%">
              <feDropShadow dx="0" dy="0" stdDeviation="7" floodColor={C.accent} floodOpacity="0.28" />
            </filter>
            <filter id="lg" x="-50%" y="-50%" width="200%" height="200%">
              <feDropShadow dx="0" dy="0" stdDeviation="5" floodColor={C.accent} floodOpacity="0.15" />
            </filter>
            <filter id="path-glow" x="-60%" y="-60%" width="220%" height="220%">
              <feDropShadow dx="0" dy="0" stdDeviation="8" floodColor={PATH_COLOR} floodOpacity="0.7" />
            </filter>
          </defs>
          {allNodes.map(node => {
            if (node.type !== "split") return null;
            const show = visibleSet.has(node.id);
            const edgeLOnPath = samplePath.has(node.id) && samplePath.has(node.left?.id);
            const edgeROnPath = samplePath.has(node.id) && samplePath.has(node.right?.id);
            return (
              // Prefix key with curTree so React creates fresh DOM elements on tab
              // switch instead of reusing them with stale CSS transition state.
              <g key={`${curTree}-${node.id}`}>
                <Edge p1={positions[node.id]} p2={positions[node.left?.id]}  visible={show}
                  label={`≤ ${fmtThresh(node.threshold)}`}
                  onPath={edgeLOnPath} sampleActive={samplePath.size > 0} />
                <Edge p1={positions[node.id]} p2={positions[node.right?.id]} visible={show}
                  label={`> ${fmtThresh(node.threshold)}`}
                  onPath={edgeROnPath} sampleActive={samplePath.size > 0} />
              </g>
            );
          })}
          {allNodes.map(node => {
            const show  = visibleSet.has(node.id);
            const phase = ts.nodeId === node.id ? ts.phase : show ? 2 : 0;
            return <TreeNode key={`${curTree}-${node.id}`} node={node} show={show || ts.nodeId === node.id} phase={phase}
              pos={positions[node.id]} allClasses={classLabels}
              onPath={samplePath.has(node.id)} sampleActive={samplePath.size > 0}
              isRegression={activeTaskType === "regression"} />;
          })}
        </svg>

        {/* Floating zoom controls (bottom-left) */}
        <div style={{
          position: "absolute", bottom: 14, left: 14, zIndex: 10,
          display: "flex", flexDirection: "column", alignItems: "center", gap: 1,
          background: "rgba(10,14,23,0.92)",
          borderRadius: 12, padding: "6px 5px",
          backdropFilter: "blur(8px)",
          boxShadow: "0 8px 28px rgba(0,0,0,0.55), inset 0 0 0 1px rgba(255,255,255,0.07)",
        }}>
          {[
            { label: "+", action: () => setZoom(z => Math.min(4, z * 1.25)) },
            null, // percentage display
            { label: "−", action: () => setZoom(z => Math.max(0.08, z * 0.8)) },
          ].map((item) =>
            item === null ? (
              <div key="pct" style={{
                fontSize: 9, color: C.dim, textAlign: "center",
                padding: "2px 8px", userSelect: "none", minWidth: 36,
              }}>{Math.round(zoom * 100)}%</div>
            ) : (
              <button key={item.label} onClick={item.action} style={{
                background: "none", border: "none", cursor: "pointer",
                color: C.dim, fontSize: 14, fontFamily: "inherit",
                width: 28, height: 24, display: "flex", alignItems: "center",
                justifyContent: "center", borderRadius: 5,
                transition: "color 0.15s, background 0.15s",
              }}
              onMouseEnter={e => { e.currentTarget.style.color = C.text; e.currentTarget.style.background = C.border; }}
              onMouseLeave={e => { e.currentTarget.style.color = C.dim; e.currentTarget.style.background = "none"; }}
              >{item.label}</button>
            )
          )}
          <div style={{ height: 1, width: "80%", background: C.border, margin: "2px 0" }} />
          <button onClick={() => { setZoom(1); centerTree(curTree); }} style={{
            background: "none", border: "none", cursor: "pointer",
            color: C.dim, fontSize: 9, fontFamily: "inherit",
            padding: "3px 6px", borderRadius: 5,
            transition: "color 0.15s, background 0.15s",
          }}
          onMouseEnter={e => { e.currentTarget.style.color = C.text; e.currentTarget.style.background = C.border; }}
          onMouseLeave={e => { e.currentTarget.style.color = C.dim; e.currentTarget.style.background = "none"; }}
          >Fit</button>
        </div>
      </div>
      {/* end SVG canvas */}

      {/* ── Calculations sidebar ──────────────────────────────────────────── */}
      <div data-tutorial="calculations-panel" style={{
        width: calcExpanded ? calcPanelWidth : 28,
        flexShrink: 0,
        borderLeft: "1px solid rgba(255,255,255,0.05)",
        background: "#080c14",
        position: "relative",
        display: "flex",
        flexDirection: "column",
        overflow: "hidden",
      }}>
        {/* Drag handle / toggle — left edge of sidebar */}
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
            /* Grip dots */
            <div style={{ display: "flex", flexDirection: "column", gap: 3, pointerEvents: "none" }}>
              {[0,1,2].map(i => (
                <div key={i} style={{ width: 2, height: 2, borderRadius: "50%", background: C.dim, opacity: 0.5 }} />
              ))}
            </div>
          ) : (
            <span style={{ fontSize: 8, color: C.dim, pointerEvents: "none" }}>‹</span>
          )}
        </div>

        {/* Collapsed state: rotated label */}
        {!calcExpanded && (
          <div style={{
            position: "absolute", inset: 0, display: "flex",
            alignItems: "center", justifyContent: "center", pointerEvents: "none",
          }}>
            <span style={{
              fontSize: 8.5, color: C.muted, fontWeight: 500,
              transform: "rotate(-90deg)", whiteSpace: "nowrap", letterSpacing: "0.08em",
            }}>Calculations</span>
          </div>
        )}

        {/* Expanded content */}
        {calcExpanded && (
          <div style={{
            flex: 1, minHeight: 0, overflowY: "auto", overflowX: "hidden",
            paddingLeft: 12, /* leave room for the 10px drag handle */
            scrollbarWidth: "thin", scrollbarColor: `${C.dimmer} transparent`,
          }}>
            <div style={{ fontSize: 9, color: C.muted, fontWeight: 500, padding: "10px 10px 6px 6px" }}>Calculations</div>

            {/* Empty state */}
            {!currentNode && (
              <div style={{ padding: "12px 10px 12px 6px", color: C.dim, fontSize: 9.5 }}>
                Press <span style={{ color: C.muted }}>▶ Grow</span> or use <span style={{ color: C.muted }}>→</span> to begin…
              </div>
            )}

            {/* ── Leaf node ──────────────────────────────────────────── */}
            {currentNode?.type === "leaf" && (() => {

              // Regression leaf
              if (currentNode.mean !== undefined) {
                return (
                  <div style={{ padding: "10px 10px 12px 6px" }}>
                    <div style={{ fontSize: 8.5, color: C.muted, marginBottom: 8 }}>Leaf · Prediction</div>
                    <div style={{ display: "flex", gap: 12, flexWrap: "wrap", marginBottom: 10 }}>
                      <div>
                        <div style={{ fontSize: 8, color: C.muted, marginBottom: 2 }}>Mean</div>
                        <div style={{ fontSize: 13, fontWeight: 700, color: C.blue, fontFamily: "'JetBrains Mono',monospace" }}>
                          {formatRegVal(currentNode.mean)}
                        </div>
                      </div>
                      <div>
                        <div style={{ fontSize: 8, color: C.muted, marginBottom: 2 }}>Variance</div>
                        <div style={{ fontSize: 10, color: C.text, fontFamily: "'JetBrains Mono',monospace" }}>
                          {currentNode.variance.toFixed(3)}
                        </div>
                      </div>
                      <div>
                        <div style={{ fontSize: 8, color: C.muted, marginBottom: 2 }}>n</div>
                        <div style={{ fontSize: 10, color: C.text, fontFamily: "'JetBrains Mono',monospace" }}>
                          {currentNode.samples}
                        </div>
                      </div>
                    </div>
                    <div style={{ fontSize: 8.5, color: C.muted }}>
                      Range: <span style={{ color: C.text }}>{formatRegVal(currentNode.min)} – {formatRegVal(currentNode.max)}</span>
                    </div>
                  </div>
                );
              }

              // Classification leaf
              const predColor    = classColor(currentNode.prediction, classLabels);
              const sortedCounts = Object.entries(currentNode.classCounts ?? {}).sort((a, b) => b[1] - a[1]);
              const total = currentNode.samples;
              let xCursor = 0;
              const barSegs = classLabels.map(cls => {
                const cnt = currentNode.classCounts?.[cls] ?? 0;
                const pct = total > 0 ? (cnt / total) * 100 : 0;
                const seg = { cls, cnt, pct, color: classColor(cls, classLabels), x: xCursor };
                xCursor += pct;
                return seg;
              }).filter(s => s.pct > 0);
              return (
                <div style={{ padding: "10px 10px 12px 6px" }}>
                  <div style={{ fontSize: 8.5, color: C.muted, marginBottom: 8 }}>Leaf · Prediction</div>
                  <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 10 }}>
                    <div style={{ fontSize: 13, fontWeight: 700, color: predColor, fontFamily: "'JetBrains Mono',monospace" }}>
                      {currentNode.prediction}
                    </div>
                  </div>
                  <div style={{ fontSize: 8.5, color: C.muted, marginBottom: 6 }}>
                    Gini <span style={{ color: C.text, fontFamily: "'JetBrains Mono',monospace" }}>{currentNode.impurity.toFixed(4)}</span>
                    {"  "}n = <span style={{ color: C.text, fontFamily: "'JetBrains Mono',monospace" }}>{currentNode.samples}</span>
                  </div>
                  <div style={{ height: 6, borderRadius: 3, overflow: "hidden", background: "rgba(255,255,255,0.05)", marginBottom: 6, display: "flex" }}>
                    {barSegs.map(s => (
                      <div key={s.cls} style={{ width: `${s.pct}%`, background: s.color, transition: "width .3s" }} />
                    ))}
                  </div>
                  <div style={{ display: "flex", flexWrap: "wrap", gap: "3px 10px" }}>
                    {sortedCounts.map(([cls, cnt]) => (
                      <div key={cls} style={{ display: "flex", alignItems: "center", gap: 4, fontSize: 8.5 }}>
                        <div style={{ width: 6, height: 6, borderRadius: 1.5, background: classColor(cls, classLabels), flexShrink: 0 }} />
                        <span style={{ color: C.dim }}>{cls}</span>
                        <span style={{ color: C.text, fontFamily: "'JetBrains Mono',monospace" }}>{cnt}</span>
                      </div>
                    ))}
                  </div>
                </div>
              );
            })()}

            {/* ── Split node ────────────────────────────────────────── */}
            {currentNode?.type === "split" && (() => {
              const candidateEvals = (currentNode.allFeatureEvals ?? [])
                .filter(ev => currentNode.candidateIndices.includes(ev.featureIndex))
                .sort((a, b) => a.gini - b.gini);
              const maxGini = Math.max(...candidateEvals.map(e => e.gini), 0.001);

              return (<>
                {/* ① Node header */}
                <div style={{ padding: "10px 10px 8px 6px" }}>
                  <div style={{ fontSize: 9, color: C.text, fontWeight: 600, marginBottom: 4 }}>
                    Depth {currentNode.depth} · {currentNode.samples} samples
                  </div>
                  {ts.phase === 0 && (
                    <div style={{ display: "flex", alignItems: "center", gap: 6, marginTop: 6 }}>
                      <div style={{ display: "flex", gap: 2, alignItems: "center" }}>
                        {[0,1,2].map(i => (
                          <div key={i} style={{
                            width: 4, height: 4, borderRadius: "50%",
                            background: C.accent, opacity: 0.3 + i * 0.3,
                            animation: `growPulse ${1 + i * 0.2}s ease-in-out infinite`,
                          }} />
                        ))}
                      </div>
                      <span style={{ fontSize: 8.5, color: C.dim }}>
                        Sampling {currentNode.candidateIndices?.length ?? "?"} of {activeFeatures.length}…
                      </span>
                    </div>
                  )}
                  {ts.phase >= 1 && (() => {
                    const names = currentNode.candidateIndices.map(i => activeFeatures[i]);
                    const MAX = 5;
                    const shown = names.slice(0, MAX);
                    const extra = names.length - MAX;
                    return (
                      <div style={{ marginTop: 4, fontSize: 8.5, color: C.dim, lineHeight: 1.55, wordBreak: "break-word" }}>
                        Evaluating{" "}
                        <span style={{ color: `${C.accent}cc` }}>
                          {shown.join(", ")}
                          {extra > 0 && <span style={{ color: C.dim }}>{` … and ${extra} more`}</span>}
                        </span>
                      </div>
                    );
                  })()}
                </div>

                {/* ② Gini bar chart — phase 1+ */}
                {ts.phase >= 1 && (
                  <div style={{ borderTop: "1px solid rgba(255,255,255,0.05)", padding: "10px 10px 10px 6px" }}>
                    <div style={{ fontSize: 8.5, color: C.dim, marginBottom: 7 }}>
                      {activeTaskType === "regression" ? "MSE" : "Gini"} per candidate{" "}
                      <span style={{ color: C.muted }}>(shorter = better)</span>
                    </div>
                    {/*
                      CSS Grid: 3 columns — name (auto), bar (1fr), value (40px).
                      "auto" sizes to the widest name, giving names priority for space.
                      "1fr" fills whatever remains, so bars grow only after all names fit.
                      All 3 cells per row are direct grid children so bars column-align.
                    */}
                    <div style={{
                      display: "grid",
                      gridTemplateColumns: "auto minmax(20px, 1fr) 40px",
                      columnGap: 8,
                      rowGap: 5,
                      alignItems: "center",
                      overflow: "hidden",
                    }}>
                      {candidateEvals.map((ev, j) => {
                        const isChosen = ts.phase >= 2 && ev.featureIndex === currentNode.featureIndex;
                        const barPct = maxGini > 0 ? (ev.gini / maxGini) * 100 : 50;
                        const rowColor = isChosen ? C.green : C.dim;
                        const fadeStyle = {
                          opacity: 0,
                          animation: `taxFadeIn 0.25s ease ${j * 0.06}s forwards`,
                        };
                        return [
                          /* Feature name cell */
                          <div key={`n${ev.featureIndex}`} style={{
                            ...fadeStyle,
                            fontSize: 8.5, color: rowColor, fontWeight: isChosen ? 600 : 400,
                            whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis",
                            minWidth: 0, transition: "color .2s",
                          }} title={activeFeatures[ev.featureIndex]}>
                            {activeFeatures[ev.featureIndex]}
                          </div>,
                          /* Bar cell */
                          <div key={`b${ev.featureIndex}`} style={{
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
                          <div key={`v${ev.featureIndex}`} style={{
                            ...fadeStyle,
                            fontSize: 8, color: rowColor, fontFamily: "'JetBrains Mono',monospace",
                            textAlign: "right", whiteSpace: "nowrap", transition: "color .2s",
                          }}>
                            {ev.gini.toFixed(3)}{isChosen && <span style={{ marginLeft: 3, fontSize: 7 }}>✓</span>}
                          </div>,
                        ];
                      })}
                    </div>
                  </div>
                )}

                {/* ③ Global-best warning — phase 2 */}
                {ts.phase >= 2 && currentNode.globalBestIdx !== currentNode.featureIndex && (
                  <div style={{
                    borderTop: "1px solid rgba(255,255,255,0.05)",
                    margin: "0 10px 8px 6px",
                    padding: "8px 10px",
                    borderRadius: 7,
                    background: `${C.accent}0a`,
                    borderLeft: `3px solid ${C.accent}88`,
                    fontSize: 8.5, color: C.dim, lineHeight: 1.6,
                  }}>
                    <span style={{ color: `${C.accent}cc`, fontWeight: 600 }}>Subset missed global best</span>
                    <br />
                    <span style={{ fontFamily: "'JetBrains Mono',monospace", color: C.text, wordBreak: "break-word" }}>
                      {activeFeatures[currentNode.globalBestIdx]}
                    </span>{" "}
                    had {activeTaskType === "regression" ? "MSE" : "G"}={currentNode.globalBestGini.toFixed(4)} but wasn't in subset
                  </div>
                )}

                {/* ④ Split result card — phase 2 */}
                {ts.phase >= 2 && (
                  <div style={{ borderTop: "1px solid rgba(255,255,255,0.05)", padding: "10px 10px 12px 6px" }}>
                    <div style={{ fontSize: 8.5, color: C.dim, marginBottom: 5 }}>Split decision</div>
                    <div style={{
                      padding: "8px 10px", borderRadius: 7,
                      background: `${C.green}0f`,
                      borderLeft: `3px solid ${C.green}88`,
                    }}>
                      <div style={{
                        fontSize: 10, fontWeight: 700, color: C.green,
                        fontFamily: "'JetBrains Mono',monospace",
                        wordBreak: "break-all", marginBottom: 3,
                      }} title={currentNode.featureName}>
                        {currentNode.featureName}
                      </div>
                      <div style={{ fontSize: 9.5, color: C.dim, fontFamily: "'JetBrains Mono',monospace" }}>
                        ≤ {currentNode.threshold}
                      </div>
                      <div style={{ fontSize: 8.5, color: C.muted, marginTop: 2 }}>
                        {activeTaskType === "regression" ? "MSE" : "G"}={currentNode.gini.toFixed(4)}
                      </div>
                    </div>
                  </div>
                )}
              </>);
            })()}
          </div>
        )}
      </div>
      {/* end canvas row flex */}
      </div>

      {/* Feature pool */}
      <div data-tutorial="feature-pool" style={{ padding: "12px 16px 6px" }}>
        <div style={{
          background: "#0c1018", borderRadius: 14,
          boxShadow: "0 2px 16px rgba(0,0,0,0.35), inset 0 0 0 1px rgba(255,255,255,0.04)",
          padding: "14px 18px",
        }}>
          {(() => {
            const showTrueBestWarning = ts.phase >= 2
              && currentNode?.type === "split"
              && currentNode?.globalBestIdx !== currentNode?.featureIndex;
            return (
              <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 10, flexWrap: "wrap" }}>
                <span style={{ fontSize: 9, color: C.muted, fontWeight: 400 }}>Feature pool</span>
                <span style={{ fontSize: 8.5 }}>
                  <span style={{ color: C.dim }}>● not sampled</span>{"  "}
                  <span style={{ color: `${C.accent}bb` }}>● candidate</span>{"  "}
                  <span style={{ color: C.green }}>● chosen</span>{"  "}
                  <span style={{ color: C.leafB }}>● true best</span>
                </span>
                {showTrueBestWarning && (
                  <span style={{ fontSize: 8, color: C.leafB, fontWeight: 600 }}>
                    ⚠ true best excluded from subset
                  </span>
                )}
              </div>
            );
          })()}
          <div style={{ display: "flex", flexWrap: "wrap", gap: 5, justifyContent: "center" }}>
          {activeFeatures.map((f, i) => {
            const cn          = currentNode?.type === "split" ? currentNode : null;
            const isCand      = cn?.candidateIndices?.includes(i);
            const isBest      = cn && i === cn.featureIndex;
            const isGlobalBest= cn && i === cn.globalBestIdx && cn.globalBestIdx !== cn.featureIndex;
            const ev          = cn?.allFeatureEvals?.find(e => e.featureIndex === i);
            const showGini    = ts.phase >= 1 && cn && ev;
            let bg, col, shd = "none";
            if (ts.phase >= 2 && isBest) {
              bg = `${C.green}22`; col = C.green; shd = `0 0 14px ${C.green}22`;
            } else if (ts.phase >= 2 && isGlobalBest) {
              bg = `${C.leafB}15`; col = C.leafB;
              shd = `0 0 0 1px ${C.leafB}55, 0 0 10px ${C.leafB}18`;
            } else if (ts.phase >= 1 && isCand) {
              bg = `${C.accent}12`; col = `${C.accent}bb`;
            } else {
              bg = "rgba(255,255,255,0.03)"; col = C.dim;
            }
            const giniCol = (ts.phase >= 2 && isBest)       ? C.green
                          : (ts.phase >= 2 && isGlobalBest)  ? C.leafB
                          : (ts.phase >= 1 && isCand)        ? `${C.accent}bb`
                          : C.dim;
            return (
              <div key={i} style={{
                padding: "5px 11px", borderRadius: 10, fontSize: 10, fontFamily: "'JetBrains Mono',monospace",
                fontWeight: (isBest && ts.phase >= 2) ? 600 : 400,
                background: bg, color: col, boxShadow: shd,
                transition: "all .3s ease-out", minWidth: 80, textAlign: "center",
              }}>
                {f}
                {showGini && (
                  <div style={{ fontSize: 7.5, marginTop: 1, color: giniCol }}>
                    {activeTaskType === "regression" ? "MSE" : "G"}={ev.gini.toFixed(3)}
                    {isGlobalBest && ts.phase >= 2 && (
                      <span style={{ marginLeft: 3, fontSize: 6.5, opacity: 0.85 }}>← true best</span>
                    )}
                  </div>
                )}
              </div>
            );
          })}
          </div>
        </div>
      </div>

      {/* ── Scroll-down hint — fades in when all trees done ─────────────────── */}
      <div style={{
        height: 32, display: "flex", alignItems: "center", justifyContent: "center",
        opacity: scrollHint ? 1 : 0,
        transition: "opacity 0.6s ease",
        pointerEvents: "none",
      }}>
        <span style={{ fontSize: 10, color: C.accent, fontWeight: 600, letterSpacing: "0.05em" }}>
          Scroll down to predict ↓
        </span>
      </div>

      {/* ── Section gap ──────────────────────────────────────────────────────── */}
      <div style={{ height: 24 }} />

      {/* ── Prediction section break ─────────────────────────────────────────── */}
      <div ref={predSectionRef} style={{ padding: "0 20px" }}>
        <div style={{
          height: 1,
          background: `linear-gradient(to right, transparent, ${PATH_COLOR}40 25%, ${PATH_COLOR}28 75%, transparent)`,
        }} />
      </div>

      {/* ── Prediction section header (sticky) ───────────────────────────────── */}
      <div style={{
        position: "sticky",
        top: 0,
        zIndex: 8,
        background: C.bg,
        padding: "10px 20px 8px",
        boxShadow: "0 4px 20px rgba(0,0,0,0.5)",
        display: "flex",
        alignItems: "center",
        gap: 12,
      }}>
        <span style={{ fontSize: 11, fontWeight: 700, color: PATH_COLOR, letterSpacing: "0.06em", flexShrink: 0 }}>
          Step 2
        </span>
        <span style={{ fontSize: 11, fontWeight: 700, color: C.text, letterSpacing: "0.04em", flexShrink: 0 }}>
          Prediction
        </span>
        <div style={{ flex: 1, height: 1, background: `linear-gradient(to right, ${PATH_COLOR}40, transparent)` }} />
        <span style={{ fontSize: 9, color: C.muted, flexShrink: 0 }}>
          {completedTrees.length}/{nEstimators} trees ready
        </span>
      </div>

      {/* ── Prediction Panel ──────────────────────────────────────────────────── */}
      <div style={{ padding: "0 16px 18px" }}>
        <div style={{
          padding: "14px 16px", background: "#0c1018", borderRadius: 14,
          boxShadow: "0 2px 16px rgba(0,0,0,0.35), inset 0 0 0 1px rgba(255,255,255,0.04)",
        }}>
          {!hasEnsemble ? (
            <div style={{ fontSize: 10.5, color: C.dim, padding: "2px 0" }}>
              Grow your trees first to start making predictions.
            </div>
          ) : (<>

            {/* ↑ View path in tree — appears when a sample is selected */}
            {selectedSample && (
              <div style={{ display: "flex", justifyContent: "flex-end", marginBottom: 8, marginTop: -2 }}>
                <button
                  onClick={() => canvasRef.current?.scrollIntoView({ behavior: "smooth", block: "start" })}
                  style={{
                    padding: "5px 13px", borderRadius: 20,
                    background: "rgba(255,255,255,0.05)",
                    border: `1px solid rgba(255,255,255,0.1)`,
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

            {/* ── Sample selector ───────────────────────────────────────────── */}
            <div data-tutorial="sample-selector" style={{ display: "flex", alignItems: "center", gap: 12, marginBottom: 16 }}>
              <div style={{
                position: "relative", display: "flex", alignItems: "center",
                borderRadius: 16, background: "rgba(255,255,255,0.04)",
                boxShadow: "0 8px 32px rgba(0,0,0,0.4), 0 0 0 1px rgba(255,255,255,0.08)",
                maxWidth: 500, flex: 1,
              }}>
                <div style={{ padding: "0 14px 0 18px", color: C.dim, fontSize: 16, flexShrink: 0 }}>⌕</div>
                <select
                  value={safeSampleIdx ?? ""}
                  onChange={e => setSelectedSampleIdx(e.target.value === "" ? null : +e.target.value)}
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

            {/* ── No sample selected ──────────────────────────────────────────── */}
            {!selectedSample && (
              <div style={{ fontSize: 10.5, color: C.dim, paddingBottom: 4, textAlign: "center", lineHeight: 1.7 }}>
                Select a sample above to see how each tree predicts it and follow its path through the forest.
              </div>
            )}

            {/* ── Per-tree vote cards + ensemble ──────────────────────────────── */}
            {selectedSample && (() => {
              // Shared tree card renderer
              const treeCards = (
                <div style={{ display: "flex", gap: 5, flexWrap: "wrap", alignItems: "center" }}>
                  {trees.map((_, i) => {
                    const spt      = sampleTreePreds.find(t => t.idx === i);
                    const hasPred  = spt !== undefined;
                    const isActive = i === curTree;

                    if (activeTaskType === "regression") {
                      const predStr = hasPred ? formatRegVal(spt.samplePred) : "…";
                      return (
                        <button key={i} onClick={() => setCurTree(i)}
                          title={`T${i + 1}: ${hasPred ? predStr : "not yet grown"}`}
                          style={{
                            minWidth: 46, height: 46, borderRadius: 9, border: "none",
                            display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center",
                            padding: "0 7px", cursor: "pointer",
                            background: hasPred ? `${C.blue}18` : "rgba(255,255,255,0.03)",
                            boxShadow: isActive
                              ? `inset 0 0 0 2px ${hasPred ? C.blue : C.dimmer}`
                              : hasPred ? `inset 0 0 0 1px ${C.blue}44` : "inset 0 0 0 1px rgba(255,255,255,0.06)",
                            transition: "box-shadow .2s, background .2s",
                            fontFamily: "'JetBrains Mono',monospace",
                          }}
                          onMouseEnter={e => { e.currentTarget.style.background = hasPred ? `${C.blue}28` : "rgba(255,255,255,0.07)"; }}
                          onMouseLeave={e => { e.currentTarget.style.background = hasPred ? `${C.blue}18` : "rgba(255,255,255,0.03)"; }}>
                          <div style={{ fontSize: 7, color: isActive ? C.text : C.dim }}>T{i + 1}</div>
                          <div style={{ fontSize: 9, fontWeight: 700, color: hasPred ? C.blue : C.dimmer, whiteSpace: "nowrap" }}>
                            {predStr}
                          </div>
                        </button>
                      );
                    }

                    const pred   = spt?.samplePred;
                    const color  = pred ? classColor(pred, classLabels) : C.dimmer;
                    const abbrev = pred && pred.length > 9 ? pred.slice(0, 8) + "…" : pred;
                    return (
                      <button key={i} onClick={() => setCurTree(i)}
                        title={`T${i + 1}: ${pred ?? "not yet grown"}`}
                        style={{
                          minWidth: 46, height: 46, borderRadius: 9, border: "none",
                          display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center",
                          padding: "0 7px", cursor: "pointer",
                          background: pred ? `${color}18` : "rgba(255,255,255,0.03)",
                          boxShadow: isActive
                            ? `inset 0 0 0 2px ${pred ? color : C.dimmer}`
                            : pred ? `inset 0 0 0 1px ${color}44` : "inset 0 0 0 1px rgba(255,255,255,0.06)",
                          transition: "box-shadow .2s, background .2s",
                          fontFamily: "'JetBrains Mono',monospace",
                        }}
                        onMouseEnter={e => { e.currentTarget.style.background = pred ? `${color}28` : "rgba(255,255,255,0.07)"; }}
                        onMouseLeave={e => { e.currentTarget.style.background = pred ? `${color}18` : "rgba(255,255,255,0.03)"; }}>
                        <div style={{ fontSize: 7, color: isActive ? C.text : C.dim }}>T{i + 1}</div>
                        <div style={{ fontSize: 9, fontWeight: 700, color: pred ? color : C.dimmer, whiteSpace: "nowrap" }}>
                          {pred ? abbrev : "…"}
                        </div>
                      </button>
                    );
                  })}
                </div>
              );

              // ── Decision Tree: original stacked layout (single tree, no ensemble) ──
              if (mode === "decision-tree") {
                return (
                  <>
                    <div style={{ marginBottom: 14 }}>{treeCards}</div>
                  </>
                );
              }

              // ── Bagging / Random Forest: ensemble card left, tree cards right ────
              const ensembleCard = (() => {
                if (activeTaskType === "regression" && sampleMean !== null) {
                  const err      = sampleTrueLabel !== null ? Math.abs(sampleMean - sampleTrueLabel) : null;
                  const relErr   = err !== null && Math.abs(sampleTrueLabel) > 0 ? err / Math.abs(sampleTrueLabel) : null;
                  const errColor = relErr !== null ? (relErr < 0.1 ? C.green : relErr < 0.25 ? C.orange : C.red) : C.muted;
                  return (
                    <div style={{
                      flexShrink: 0, width: 210, alignSelf: "stretch",
                      padding: "16px 16px", borderRadius: 14,
                      background: `linear-gradient(135deg, ${C.blue}12, rgba(10,14,23,0.6))`,
                      boxShadow: `0 0 0 1.5px ${C.blue}55, 0 4px 24px ${C.blue}18, inset 0 1px 0 ${C.blue}22`,
                      display: "flex", flexDirection: "column", justifyContent: "center",
                    }}>
                      <div style={{ fontSize: 8.5, color: C.dim, fontWeight: 500, marginBottom: 6, letterSpacing: "0.06em", textTransform: "uppercase" }}>
                        Ensemble · {completedTrees.length} tree{completedTrees.length !== 1 ? "s" : ""}
                      </div>
                      <div style={{ fontSize: 26, fontWeight: 800, color: C.blue, fontFamily: "'JetBrains Mono',monospace", lineHeight: 1, marginBottom: 10 }}>
                        {formatRegVal(sampleMean)}
                      </div>
                      {sampleTrueLabel !== null && (
                        <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
                          <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
                            <span style={{ fontSize: 8.5, color: C.muted, textTransform: "uppercase", letterSpacing: "0.05em" }}>True</span>
                            <span style={{ fontSize: 13, fontWeight: 700, color: C.text, fontFamily: "'JetBrains Mono',monospace" }}>{formatRegVal(sampleTrueLabel)}</span>
                          </div>
                          <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
                            <span style={{ fontSize: 8.5, color: C.muted, textTransform: "uppercase", letterSpacing: "0.05em" }}>Error</span>
                            <span style={{ fontSize: 13, fontWeight: 700, color: errColor, fontFamily: "'JetBrains Mono',monospace" }}>{formatRegVal(err)}</span>
                          </div>
                        </div>
                      )}
                    </div>
                  );
                }
                if (activeTaskType === "classification" && sampleMajority) {
                  const majColor = classColor(sampleMajority, classLabels);
                  return (
                    <div style={{
                      flexShrink: 0, width: 210, alignSelf: "stretch",
                      padding: "16px 16px", borderRadius: 14,
                      background: `linear-gradient(135deg, ${majColor}12, rgba(10,14,23,0.6))`,
                      boxShadow: `0 0 0 1.5px ${majColor}55, 0 4px 24px ${majColor}18, inset 0 1px 0 ${majColor}22`,
                      display: "flex", flexDirection: "column", justifyContent: "center",
                    }}>
                      <div style={{ fontSize: 8.5, color: C.dim, fontWeight: 500, marginBottom: 5, letterSpacing: "0.06em", textTransform: "uppercase" }}>
                        Ensemble · {completedTrees.length} tree{completedTrees.length !== 1 ? "s" : ""}
                      </div>
                      <div style={{ fontSize: 22, fontWeight: 800, color: majColor, lineHeight: 1, marginBottom: 10 }}>
                        {sampleMajority}
                      </div>
                      {/* Vote bar */}
                      <div style={{ marginBottom: 8 }}>
                        <div style={{ height: 5, borderRadius: 3, overflow: "hidden", background: "rgba(0,0,0,0.3)", display: "flex", marginBottom: 5 }}>
                          {classLabels.map(cls => {
                            const pct = ((sampleVotesPerClass[cls] ?? 0) / completedTrees.length) * 100;
                            return pct > 0
                              ? <div key={cls} style={{ width: `${pct}%`, background: classColor(cls, classLabels), transition: "width .3s" }} />
                              : null;
                          })}
                        </div>
                        <div style={{ display: "flex", flexDirection: "column", gap: 3 }}>
                          {classLabels.map(cls => {
                            const n = sampleVotesPerClass[cls] ?? 0;
                            if (n === 0) return null;
                            return (
                              <span key={cls} style={{ fontSize: 8.5, color: C.muted }}>
                                <strong style={{ color: classColor(cls, classLabels), fontFamily: "'JetBrains Mono',monospace" }}>{n}</strong>
                                <span style={{ color: C.dimmer }}>/{completedTrees.length} </span>
                                <span style={{ color: classColor(cls, classLabels) }}>{cls}</span>
                              </span>
                            );
                          })}
                        </div>
                      </div>
                      {sampleTrueLabel !== null && (
                        <div style={{ display: "flex", gap: 10, flexWrap: "wrap", alignItems: "center", paddingTop: 8, borderTop: "1px solid rgba(255,255,255,0.06)" }}>
                          <div style={{ display: "flex", alignItems: "center", gap: 5 }}>
                            <span style={{ fontSize: 8.5, color: C.muted, textTransform: "uppercase", letterSpacing: "0.05em" }}>True</span>
                            <span style={{ fontSize: 12, fontWeight: 700, color: classColor(sampleTrueLabel, classLabels) }}>{sampleTrueLabel}</span>
                          </div>
                          <span style={{ fontSize: 12, fontWeight: 800, color: sampleCorrect ? C.green : C.red }}>
                            {sampleCorrect ? "✓ Correct" : "✗ Wrong"}
                          </span>
                        </div>
                      )}
                    </div>
                  );
                }
                // No ensemble result yet (trees still growing)
                return (
                  <div style={{
                    flexShrink: 0, width: 210, alignSelf: "stretch",
                    padding: "16px 16px", borderRadius: 14,
                    background: "rgba(255,255,255,0.02)",
                    boxShadow: "inset 0 0 0 1px rgba(255,255,255,0.06)",
                    display: "flex", flexDirection: "column", justifyContent: "center",
                  }}>
                    <div style={{ fontSize: 8.5, color: C.dim, textTransform: "uppercase", letterSpacing: "0.06em", marginBottom: 6 }}>Ensemble</div>
                    <div style={{ fontSize: 12, color: C.dimmer }}>Complete trees to see ensemble prediction.</div>
                  </div>
                );
              })();

              return (
                <div data-tutorial="ensemble-vote" style={{ display: "flex", gap: 12, alignItems: "center", marginBottom: 14 }}>
                  {ensembleCard}
                  {treeCards}
                </div>
              );
            })()}

            {/* ── Metrics cards ───────────────────────────────────────────────── */}
            {activeTaskType === "classification" && (forestAccuracy !== null || avgOobAccuracy !== null) && (
              <div data-tutorial="accuracy-cards" style={{ display: "flex", gap: 10, marginTop: selectedSample ? 4 : 0 }}>
                {forestAccuracy !== null && (
                  <div style={{
                    flex: 1, padding: "12px 14px", borderRadius: 12,
                    background: "rgba(255,255,255,0.03)",
                    boxShadow: "inset 0 0 0 1px rgba(255,255,255,0.07)",
                  }}>
                    <div style={{ fontSize: 8, color: C.dim, fontWeight: 600, letterSpacing: "0.08em", textTransform: "uppercase", marginBottom: 6 }}>Training</div>
                    <div style={{ fontSize: 9, color: C.muted, marginBottom: 2 }}>Accuracy</div>
                    <div style={{ fontSize: 20, fontWeight: 800, color: C.text, fontFamily: "'JetBrains Mono',monospace" }}>{forestAccuracy}%</div>
                  </div>
                )}
                {avgOobAccuracy !== null && (
                  <div style={{
                    flex: 1, padding: "12px 14px", borderRadius: 12,
                    background: `${C.green}0d`,
                    boxShadow: `inset 0 0 0 1px ${C.green}44`,
                    position: "relative",
                  }}>
                    <div style={{ fontSize: 8, color: C.green, fontWeight: 600, letterSpacing: "0.08em", textTransform: "uppercase", marginBottom: 6, display: "flex", alignItems: "center", gap: 5 }}>
                      Out-of-Bag
                      <span style={{ cursor: "help", color: C.dim, fontWeight: 400 }}
                        onMouseEnter={e => setOobTooltipVisible(tooltipPosition(e.currentTarget.getBoundingClientRect(), { prefer: 'above', width: 260, height: 52 }))}
                        onMouseLeave={() => setOobTooltipVisible(null)}>ⓘ</span>
                    </div>
                    <div style={{ fontSize: 9, color: C.muted, marginBottom: 2 }}>Accuracy</div>
                    <div style={{ fontSize: 20, fontWeight: 800, color: C.green, fontFamily: "'JetBrains Mono',monospace" }}>{avgOobAccuracy}%</div>
                  </div>
                )}
              </div>
            )}
            {activeTaskType === "regression" && (forestRegMetrics !== null || avgOobR2 !== null) && (
              <div style={{ display: "flex", gap: 10, marginTop: selectedSample ? 4 : 0 }}>
                {forestRegMetrics !== null && (
                  <div style={{
                    flex: 1, padding: "12px 14px", borderRadius: 12,
                    background: "rgba(255,255,255,0.03)",
                    boxShadow: "inset 0 0 0 1px rgba(255,255,255,0.07)",
                  }}>
                    <div style={{ fontSize: 8, color: C.dim, fontWeight: 600, letterSpacing: "0.08em", textTransform: "uppercase", marginBottom: 6 }}>Training</div>
                    <div style={{ fontSize: 9, color: C.muted, marginBottom: 2 }}>R²</div>
                    <div style={{ fontSize: 20, fontWeight: 800, color: C.text, fontFamily: "'JetBrains Mono',monospace" }}>{forestRegMetrics.r2}</div>
                    <div style={{ fontSize: 9, color: C.muted, marginTop: 6 }}>RMSE <span style={{ color: C.text, fontFamily: "'JetBrains Mono',monospace", fontWeight: 700 }}>{forestRegMetrics.rmse}</span></div>
                  </div>
                )}
                {avgOobR2 !== null && (
                  <div style={{
                    flex: 1, padding: "12px 14px", borderRadius: 12,
                    background: `${C.green}0d`,
                    boxShadow: `inset 0 0 0 1px ${C.green}44`,
                    position: "relative",
                  }}>
                    <div style={{ fontSize: 8, color: C.green, fontWeight: 600, letterSpacing: "0.08em", textTransform: "uppercase", marginBottom: 6, display: "flex", alignItems: "center", gap: 5 }}>
                      Out-of-Bag
                      <span style={{ cursor: "help", color: C.dim, fontWeight: 400 }}
                        onMouseEnter={e => setOobTooltipVisible(tooltipPosition(e.currentTarget.getBoundingClientRect(), { prefer: 'above', width: 260, height: 64 }))}
                        onMouseLeave={() => setOobTooltipVisible(null)}>ⓘ</span>
                    </div>
                    <div style={{ fontSize: 9, color: C.muted, marginBottom: 2 }}>R²</div>
                    <div style={{ fontSize: 20, fontWeight: 800, color: C.green, fontFamily: "'JetBrains Mono',monospace" }}>{avgOobR2}</div>
                    <div style={{ fontSize: 9, color: C.muted, marginTop: 6 }}>RMSE <span style={{ color: C.green, fontFamily: "'JetBrains Mono',monospace", fontWeight: 700 }}>{avgOobRMSE}</span></div>
                  </div>
                )}
              </div>
            )}

          </>)}
        </div>
      </div>

      {/* Footer */}
      <div style={{ padding: "8px 16px", boxShadow: "0 -1px 0 rgba(255,255,255,0.04)", fontSize: 8.5, color: C.dim, textAlign: "center" }}>
        max_features={subsetSize}/{activeFeatures.length} · Trees: {nEstimators} · Bootstrap n={TOTAL_SAMPLES}
      </div>

      {/* Tree tab tooltip */}
      {tabTooltip && (
        <div style={{
          position: "fixed",
          left: tabTooltip.left,
          top: tabTooltip.top,
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
          Double-click to instantly complete this tree
        </div>
      )}

      {/* OOB tooltip */}
      {oobTooltipVisible && (
        <div style={{
          position: "fixed", left: oobTooltipVisible.left, top: oobTooltipVisible.top,
          width: 260, background: "#141b2d", borderRadius: 9, padding: "9px 12px",
          fontSize: 9.5, color: C.dim, lineHeight: 1.6,
          boxShadow: "0 8px 32px rgba(0,0,0,0.6), inset 0 0 0 1px rgba(255,255,255,0.08)",
          zIndex: 9999, pointerEvents: "none",
        }}>
          {avgOobAccuracy !== null
            ? "Each tree is tested on the ~37% of samples not used in its bootstrap training set. This gives an unbiased estimate of generalization error with no held-out test set needed."
            : "R² measures proportion of variance explained. 1.0 is perfect, 0.0 is no better than predicting the mean. OOB estimates this on the ~37% of samples held out from each tree's bootstrap."}
        </div>
      )}

      {/* Dataset dropdown menu — portaled to escape GlobalHeader stacking context */}
      {rfDropOpen && rfDropPos && createPortal(
        <div ref={rfMenuRef} className="rf-ds-drop-menu"
          style={{ top: rfDropPos.top, left: rfDropPos.left }}>
          <button className="rf-ds-opt"
            data-active={!customDataset && builtinDataset === "heart" ? "true" : "false"}
            onClick={() => { setRfDropOpen(false); switchToBuiltin("heart"); }}>
            {!customDataset && builtinDataset === "heart" && <span style={{ fontSize: 9 }}>✓ </span>}
            Built-in: Heart Disease (Binary classification)
          </button>
          <button className="rf-ds-opt"
            data-active={!customDataset && builtinDataset === "music" ? "true" : "false"}
            onClick={() => { setRfDropOpen(false); switchToBuiltin("music"); }}>
            {!customDataset && builtinDataset === "music" && <span style={{ fontSize: 9 }}>✓ </span>}
            Built-in: Music Genres (Multiclass classification)
          </button>
          <button className="rf-ds-opt"
            data-active={!customDataset && builtinDataset === "salary" ? "true" : "false"}
            onClick={() => { setRfDropOpen(false); switchToBuiltin("salary"); }}>
            {!customDataset && builtinDataset === "salary" && <span style={{ fontSize: 9 }}>✓ </span>}
            Built-in: Salary (Regression)
          </button>
          {customDataset && (
            <button className="rf-ds-opt" data-active="true"
              onClick={() => setRfDropOpen(false)}>
              ✓ {customDataset.name}
            </button>
          )}
          <div style={{ borderTop: "1px solid rgba(255,255,255,0.06)", margin: "4px 0" }} />
          <button className="rf-ds-opt"
            onClick={() => { setRfDropOpen(false); fileInputRef.current?.click(); }}>
            ↑ Upload CSV…
          </button>
        </div>,
        document.body
      )}

      {/* Locked-param tooltips (portaled to escape stacking contexts) */}
      {lockedParamTooltip && createPortal(
        <div style={{
          position: "fixed", top: lockedParamTooltip.top, left: lockedParamTooltip.left,
          width: 240, padding: "10px 13px", borderRadius: 10,
          background: "#1a2235",
          boxShadow: "0 8px 32px rgba(0,0,0,0.7), inset 0 0 0 1px rgba(255,255,255,0.12)",
          fontSize: 10.5, color: C.text, lineHeight: 1.65,
          fontWeight: 400, zIndex: 9999, pointerEvents: "none",
          animation: "fadeInUp 0.14s ease-out",
          whiteSpace: "normal",
        }}>
          {lockedParamTooltip.which === "maxFeatures"
            ? (mode === "decision-tree"
                ? "A decision tree evaluates all features at each split — switch to Random Forest for feature subsampling"
                : "Bagging typically considers all features at each split — switch to Random Forest to enable feature subsampling")
            : "A decision tree is a single model — switch to an ensemble"}
        </div>,
        document.body
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
          Click to reset current tree · Double-click to reset all
        </div>
      )}
    </div>
  );
}
