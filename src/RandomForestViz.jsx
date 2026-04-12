import { useState, useEffect, useRef, useCallback } from "react";

const FEATURES = [
  "Age", "Income", "Education", "Hours/Week", "Experience",
  "Debt Ratio", "Credit Score", "Dependents", "Tenure", "Savings"
];

const FEATURE_SUBSET_OPTIONS = {
  sqrt: { label: "√p (sqrt)", fn: (p) => Math.max(1, Math.round(Math.sqrt(p))) },
  log2: { label: "log₂(p)", fn: (p) => Math.max(1, Math.round(Math.log2(p))) },
  half: { label: "p/2", fn: (p) => Math.max(1, Math.round(p / 2)) },
  all:  { label: "p (all)", fn: (p) => p },
};

function shuffleArray(arr) {
  const a = [...arr];
  for (let i = a.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [a[i], a[j]] = [a[j], a[i]];
  }
  return a;
}

function giniImpurity(left, right) {
  const tL = left[0] + left[1], tR = right[0] + right[1];
  if (tL === 0 || tR === 0) return 1;
  const gL = 1 - (left[0]/tL)**2 - (left[1]/tL)**2;
  const gR = 1 - (right[0]/tR)**2 - (right[1]/tR)**2;
  const t = tL + tR;
  return (tL/t) * gL + (tR/t) * gR;
}

function evalFeature(fi, parentSamples) {
  const threshold = +(Math.random() * 0.6 + 0.2).toFixed(2);
  const lA = Math.floor(Math.random() * parentSamples * 0.5);
  const lB = Math.floor(Math.random() * parentSamples * 0.3);
  const rA = Math.floor(Math.random() * (parentSamples - lA - lB) * 0.5);
  const rB = parentSamples - lA - lB - rA;
  const g = giniImpurity([lA, lB], [Math.max(0, rA), Math.max(0, rB)]);
  return { featureIndex: fi, gini: g, threshold };
}

function buildTreeData(maxDepth, subsetSize, totalSamples = 200, parentSamples = null, depth = 0) {
  if (parentSamples === null) parentSamples = totalSamples;
  if (depth >= maxDepth || parentSamples <= 5) {
    const classA = Math.floor(Math.random() * parentSamples);
    const classB = parentSamples - classA;
    const total = classA + classB;
    return {
      type: "leaf", samples: parentSamples, classA, classB,
      impurity: total > 0 ? 1 - (classA/total)**2 - (classB/total)**2 : 0,
      prediction: classA >= classB ? "A" : "B", depth,
    };
  }

  // evaluate ALL features
  const allIdx = FEATURES.map((_, i) => i);
  const allFeatureEvals = allIdx.map(fi => evalFeature(fi, parentSamples));

  // pick random subset
  const candidateIndices = shuffleArray(allIdx).slice(0, subsetSize);

  // find best within subset
  let bestFeature = candidateIndices[0], bestGini = 1, bestThreshold = 0.5;
  const candidateEvals = allFeatureEvals.filter(e => candidateIndices.includes(e.featureIndex));
  candidateEvals.forEach(ev => {
    if (ev.gini < bestGini) { bestGini = ev.gini; bestFeature = ev.featureIndex; bestThreshold = ev.threshold; }
  });

  // find global best (for comparison)
  let globalBestIdx = 0, globalBestGini = 1;
  allFeatureEvals.forEach(ev => {
    if (ev.gini < globalBestGini) { globalBestGini = ev.gini; globalBestIdx = ev.featureIndex; }
  });

  const leftSamples = Math.floor(parentSamples * (0.3 + Math.random() * 0.4));
  return {
    type: "split", featureIndex: bestFeature, featureName: FEATURES[bestFeature],
    threshold: bestThreshold, gini: bestGini, samples: parentSamples,
    candidateIndices, allFeatureEvals,
    globalBestIdx, globalBestGini,
    depth,
    left: buildTreeData(maxDepth, subsetSize, totalSamples, leftSamples, depth + 1),
    right: buildTreeData(maxDepth, subsetSize, totalSamples, parentSamples - leftSamples, depth + 1),
  };
}

function flattenNodes(node, id = "0") {
  node.id = id;
  const nodes = [node];
  if (node.type === "split") {
    nodes.push(...flattenNodes(node.left, id + "L"));
    nodes.push(...flattenNodes(node.right, id + "R"));
  }
  return nodes;
}

function getTreePrediction(node) {
  // majority class at all leaves
  const leaves = flattenNodes(node).filter(n => n.type === "leaf");
  let totalA = 0, totalB = 0;
  leaves.forEach(l => { totalA += l.classA; totalB += l.classB; });
  return totalA >= totalB ? "A" : "B";
}

function computePositions(node, width) {
  const pos = {};
  const Y_GAP = 82;
  function layout(n, xMin, xMax, d) {
    pos[n.id] = { x: (xMin + xMax) / 2, y: 40 + d * Y_GAP };
    if (n.type === "split") {
      layout(n.left, xMin, (xMin + xMax) / 2, d + 1);
      layout(n.right, (xMin + xMax) / 2, xMax, d + 1);
    }
  }
  layout(node, 20, width - 20, 0);
  return pos;
}

const C = {
  bg: "#0a0e17", panel: "#111827", border: "#1e293b",
  text: "#e2e8f0", dim: "#64748b", dimmer: "#3d4a5c",
  accent: "#f59e0b", accentG: "#f59e0b33",
  green: "#10b981", greenG: "#10b98133",
  orange: "#fb923c", red: "#ef4444", blue: "#3b82f6", leafB: "#f43f5e",
  edge: "#334155", purple: "#a78bfa",
};

function Edge({ p1, p2, visible, label }) {
  if (!visible || !p1 || !p2) return null;
  return (
    <g style={{ opacity: 1, transition: "opacity .3s" }}>
      <line x1={p1.x} y1={p1.y + 22} x2={p2.x} y2={p2.y - 24}
        stroke={C.edge} strokeWidth={1.4} strokeDasharray="4 2" />
      <text x={(p1.x+p2.x)/2} y={(p1.y+22+p2.y-24)/2-4} textAnchor="middle"
        fill={C.dim} fontSize={8} fontFamily="'JetBrains Mono',monospace">{label}</text>
    </g>
  );
}

function TreeNode({ node, show, phase, pos }) {
  if (!show || !pos) return null;
  const { x, y } = pos;
  const vis = phase >= 1 ? 1 : 0;
  if (node.type === "leaf") {
    const r = node.samples > 0 ? node.classA / node.samples : 0.5;
    return (
      <g style={{ opacity: vis, transition: "opacity .5s" }}>
        <rect x={x-38} y={y-24} width={76} height={50} rx={7}
          fill={C.panel} stroke={node.prediction==="A"?C.blue:C.leafB} strokeWidth={1.8} filter="url(#lg)" />
        <text x={x} y={y-9} textAnchor="middle" fill={C.text} fontSize={9}
          fontFamily="'JetBrains Mono',monospace" fontWeight={600}>
          {node.prediction==="A"?"Class A":"Class B"}</text>
        <rect x={x-28} y={y+2} width={56} height={5} rx={2.5} fill={C.leafB} opacity={.35}/>
        <rect x={x-28} y={y+2} width={Math.max(0,56*r)} height={5} rx={2.5} fill={C.blue}/>
        <text x={x} y={y+17} textAnchor="middle" fill={C.dim} fontSize={7.5}
          fontFamily="'JetBrains Mono',monospace">n={node.samples} G={node.impurity.toFixed(3)}</text>
      </g>
    );
  }
  return (
    <g style={{ opacity: vis, transition: "opacity .4s" }}>
      <rect x={x-52} y={y-20} width={104} height={40} rx={5}
        fill={C.panel} stroke={C.accent} strokeWidth={1.4} filter="url(#ng)" />
      <text x={x} y={y-4} textAnchor="middle" fill={C.accent} fontSize={9.5}
        fontFamily="'JetBrains Mono',monospace" fontWeight={700}>{node.featureName}</text>
      <text x={x} y={y+9} textAnchor="middle" fill={C.dim} fontSize={7.5}
        fontFamily="'JetBrains Mono',monospace">≤{node.threshold} G={node.gini.toFixed(3)} n={node.samples}</text>
    </g>
  );
}

const EMPTY_TS = { visibleIds: [], nodeId: null, phase: 0, stepIdx: -1 };

export default function RandomForestViz() {
  const [maxDepth, setMaxDepth] = useState(3);
  const [featureSubset, setFeatureSubset] = useState("sqrt");
  const [nEstimators, setNEstimators] = useState(3);
  const [trees, setTrees] = useState([]);
  const [bootstrapInfo, setBootstrapInfo] = useState([]);
  const [curTree, setCurTree] = useState(0);
  const [treeStates, setTreeStates] = useState({});
  const [growing, setGrowing] = useState(false);
  const [speed, setSpeed] = useState(1);
  const growRef = useRef(false);
  const cancelRef = useRef(false);

  const [zoom, setZoom] = useState(1);
  const [pan, setPan] = useState({ x: 0, y: 0 });
  const isDragging = useRef(false);
  const dragStart = useRef({ x: 0, y: 0, px: 0, py: 0 });
  const canvasRef = useRef(null);

  const subsetSize = FEATURE_SUBSET_OPTIONS[featureSubset].fn(FEATURES.length);
  const treeWidth = Math.max(600, Math.pow(2, maxDepth) * 120);
  const svgH = (maxDepth + 1) * 82 + 60;

  const getTS = (idx) => treeStates[idx] || EMPTY_TS;

  const setTS = useCallback((idx, patch) => {
    setTreeStates(prev => {
      const old = prev[idx] || EMPTY_TS;
      return { ...prev, [idx]: { ...old, ...patch } };
    });
  }, []);

  const TOTAL_SAMPLES = 200;

  const buildForest = useCallback(() => {
    cancelRef.current = true;
    growRef.current = false;
    setGrowing(false);
    const f = [];
    const bInfo = [];
    for (let i = 0; i < nEstimators; i++) {
      // bootstrap: sample with replacement
      const bootstrapN = TOTAL_SAMPLES;
      const drawn = new Set();
      for (let j = 0; j < bootstrapN; j++) drawn.add(Math.floor(Math.random() * TOTAL_SAMPLES));
      const inBag = drawn.size;
      const oob = TOTAL_SAMPLES - inBag;
      const oobAccuracy = +(0.55 + Math.random() * 0.35).toFixed(3);
      bInfo.push({ inBag, oob, oobAccuracy, bootstrapN });

      const t = buildTreeData(maxDepth, subsetSize, bootstrapN);
      flattenNodes(t);
      f.push(t);
    }
    setTrees(f);
    setBootstrapInfo(bInfo);
    setCurTree(0);
    setTreeStates({});
    setZoom(1);
    setPan({ x: 0, y: 0 });
  }, [maxDepth, subsetSize, nEstimators]);

  useEffect(() => { buildForest(); }, []);

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
    if (steps.length === 0) return;
    if (targetIdx < -1) targetIdx = -1;
    if (targetIdx >= steps.length) targetIdx = steps.length - 1;
    if (targetIdx === -1) {
      setTS(treeIdx, { visibleIds: [], nodeId: null, phase: 0, stepIdx: -1 });
      return;
    }
    const vis = [];
    for (let i = 0; i <= targetIdx; i++) {
      if (steps[i].commit && !vis.includes(steps[i].nodeId)) vis.push(steps[i].nodeId);
    }
    const s = steps[targetIdx];
    setTS(treeIdx, { visibleIds: vis, nodeId: s.nodeId, phase: s.phase, stepIdx: targetIdx });
  }, [getSteps, setTS]);

  const autoGrow = useCallback(async (treeIdx) => {
    if (growRef.current) return;
    growRef.current = true;
    cancelRef.current = false;
    setGrowing(true);
    setTS(treeIdx, EMPTY_TS);
    await new Promise(r => setTimeout(r, 80));
    const steps = getSteps(treeIdx);
    for (let i = 0; i < steps.length; i++) {
      if (cancelRef.current) break;
      const vis = [];
      for (let j = 0; j <= i; j++) { if (steps[j].commit && !vis.includes(steps[j].nodeId)) vis.push(steps[j].nodeId); }
      setTS(treeIdx, { visibleIds: vis, nodeId: steps[i].nodeId, phase: steps[i].phase, stepIdx: i });
      await new Promise(r => setTimeout(r, 400 / speed));
    }
    growRef.current = false;
    setGrowing(false);
  }, [getSteps, setTS, speed]);

  // instantly complete a single tree (no animation)
  const instantComplete = useCallback((treeIdx) => {
    const steps = getSteps(treeIdx);
    if (steps.length === 0) return;
    const lastIdx = steps.length - 1;
    const vis = [];
    for (let i = 0; i <= lastIdx; i++) {
      if (steps[i].commit && !vis.includes(steps[i].nodeId)) vis.push(steps[i].nodeId);
    }
    const s = steps[lastIdx];
    setTS(treeIdx, { visibleIds: vis, nodeId: s.nodeId, phase: s.phase, stepIdx: lastIdx });
  }, [getSteps, setTS]);

  // instantly complete ALL trees
  const growAllInstant = useCallback(() => {
    if (growing) return;
    for (let i = 0; i < trees.length; i++) {
      instantComplete(i);
    }
  }, [growing, trees, instantComplete]);

  useEffect(() => {
    const h = (e) => {
      if (growing) return;
      if (e.key === "ArrowRight" || e.key === "ArrowDown") {
        e.preventDefault();
        const ts = getTS(curTree);
        goToStep(curTree, ts.stepIdx + 1);
      } else if (e.key === "ArrowLeft" || e.key === "ArrowUp") {
        e.preventDefault();
        const ts = getTS(curTree);
        goToStep(curTree, ts.stepIdx - 1);
      }
    };
    window.addEventListener("keydown", h);
    return () => window.removeEventListener("keydown", h);
  });

  useEffect(() => {
    const el = canvasRef.current;
    if (!el) return;
    const h = (e) => {
      e.preventDefault();
      const rect = el.getBoundingClientRect();
      const mx = e.clientX - rect.left;
      const my = e.clientY - rect.top;

      // trackpad pinch sends ctrlKey + small deltaY
      // mouse wheel sends larger deltaY without ctrlKey
      // normalize both to a smooth continuous factor
      let delta = -e.deltaY;
      if (e.ctrlKey) {
        // trackpad pinch: deltaY is already small (~1-4), dampen further
        delta = delta * 0.008;
      } else {
        // mouse wheel: deltaY is large (~100), scale way down
        // also handle pixel vs line delta modes
        const pixelDelta = e.deltaMode === 1 ? delta * 16 : delta;
        delta = pixelDelta * 0.001;
      }

      // clamp to avoid giant jumps from fast flicks
      delta = Math.max(-0.15, Math.min(0.15, delta));
      const factor = 1 + delta;

      setZoom(z => {
        const nz = Math.max(0.08, Math.min(4, z * factor));
        const scale = nz / z;
        setPan(p => ({
          x: mx - scale * (mx - p.x),
          y: my - scale * (my - p.y),
        }));
        return nz;
      });
    };
    el.addEventListener("wheel", h, { passive: false });
    return () => el.removeEventListener("wheel", h);
  }, []);

  const onMouseDown = (e) => {
    isDragging.current = true;
    dragStart.current = { x: e.clientX, y: e.clientY, px: pan.x, py: pan.y };
  };
  const onMouseMove = (e) => {
    if (!isDragging.current) return;
    setPan({ x: dragStart.current.px + e.clientX - dragStart.current.x, y: dragStart.current.py + e.clientY - dragStart.current.y });
  };
  const onMouseUp = () => { isDragging.current = false; };

  // derived
  const currentTree = trees[curTree];
  const allNodes = currentTree ? flattenNodes(currentTree) : [];
  const positions = currentTree ? computePositions(currentTree, treeWidth) : {};
  const ts = getTS(curTree);
  const visibleSet = new Set(ts.visibleIds);
  const currentNode = allNodes.find(n => n.id === ts.nodeId);
  const totalSteps = getSteps(curTree).length;

  // ensemble vote
  const completedTrees = trees.map((t, i) => {
    const s = treeStates[i];
    const steps = getSteps(i);
    const done = s && s.stepIdx >= steps.length - 1 && steps.length > 0;
    return done ? { idx: i, prediction: getTreePrediction(t) } : null;
  }).filter(Boolean);

  const votesA = completedTrees.filter(t => t.prediction === "A").length;
  const votesB = completedTrees.filter(t => t.prediction === "B").length;
  const ensemblePrediction = votesA >= votesB ? "A" : "B";
  const hasEnsemble = completedTrees.length >= 2;

  const curBootstrap = bootstrapInfo[curTree];

  const inp = { padding: "6px 8px", borderRadius: 6, background: "#1a1f2e", border: `1px solid ${C.border}`, color: C.text, fontSize: 12, fontFamily: "'JetBrains Mono',monospace", outline: "none" };

  return (
    <div style={{ minHeight: "100vh", background: C.bg, color: C.text, fontFamily: "'JetBrains Mono','Fira Code',monospace" }}>
      {/* Header */}
      <div style={{ padding: "14px 20px 8px", borderBottom: `1px solid ${C.border}`, background: "linear-gradient(180deg,#111827,#0a0e17)" }}>
        <h1 style={{ fontSize: 18, fontWeight: 800, margin: 0, background: `linear-gradient(135deg,${C.accent},${C.green})`, WebkitBackgroundClip: "text", WebkitTextFillColor: "transparent" }}>
          Random Forest Builder
        </h1>
        <span style={{ fontSize: 10, color: C.dim }}>Scroll to zoom · Drag to pan · ←→ arrow keys to step · Double-click tree tab to instant-complete</span>
      </div>

      {/* Controls */}
      <div style={{ display: "flex", gap: 10, padding: "10px 20px", flexWrap: "wrap", alignItems: "flex-end", borderBottom: `1px solid ${C.border}` }}>
        <div style={{ display: "flex", flexDirection: "column", gap: 3 }}>
          <label style={{ fontSize: 9, color: C.dim, textTransform: "uppercase", letterSpacing: 1 }}>max_depth</label>
          <input type="number" min={1} max={5} value={maxDepth} disabled={growing}
            onChange={e => setMaxDepth(Math.max(1, Math.min(5, +e.target.value)))} style={{ ...inp, width: 54 }} />
        </div>
        <div style={{ display: "flex", flexDirection: "column", gap: 3 }}>
          <label style={{ fontSize: 9, color: C.dim, textTransform: "uppercase", letterSpacing: 1 }}>max_features</label>
          <select value={featureSubset} disabled={growing} onChange={e => setFeatureSubset(e.target.value)} style={{ ...inp, cursor: "pointer" }}>
            {Object.entries(FEATURE_SUBSET_OPTIONS).map(([k, v]) => (
              <option key={k} value={k}>{v.label} → {v.fn(FEATURES.length)}</option>
            ))}
          </select>
        </div>
        <div style={{ display: "flex", flexDirection: "column", gap: 3 }}>
          <label style={{ fontSize: 9, color: C.dim, textTransform: "uppercase", letterSpacing: 1 }}>n_estimators</label>
          <input type="number" min={1} max={100} value={nEstimators} disabled={growing}
            onChange={e => setNEstimators(Math.max(1, Math.min(100, +e.target.value)))} style={{ ...inp, width: 54 }} />
        </div>
        <div style={{ display: "flex", flexDirection: "column", gap: 3 }}>
          <label style={{ fontSize: 9, color: C.dim, textTransform: "uppercase", letterSpacing: 1 }}>speed</label>
          <select value={speed} onChange={e => setSpeed(+e.target.value)} style={{ ...inp, cursor: "pointer", width: 60 }}>
            {[0.5, 1, 2, 4].map(s => <option key={s} value={s}>{s}×</option>)}
          </select>
        </div>
        <div style={{ display: "flex", flexDirection: "column", gap: 3 }}>
          <label style={{ fontSize: 9, color: C.dim, textTransform: "uppercase", letterSpacing: 1 }}>zoom</label>
          <div style={{ display: "flex", gap: 3 }}>
            <button onClick={() => setZoom(z => Math.max(0.08, z * 0.8))} style={{ ...inp, cursor: "pointer", width: 26, textAlign: "center", padding: "5px 0" }}>−</button>
            <span style={{ ...inp, width: 38, textAlign: "center", padding: "6px 0", fontSize: 10 }}>{Math.round(zoom*100)}%</span>
            <button onClick={() => setZoom(z => Math.min(4, z * 1.25))} style={{ ...inp, cursor: "pointer", width: 26, textAlign: "center", padding: "5px 0" }}>+</button>
            <button onClick={() => { setZoom(1); setPan({ x: 0, y: 0 }); }} style={{ ...inp, cursor: "pointer", padding: "5px 6px", fontSize: 9 }}>Fit</button>
          </div>
        </div>
        <div style={{ flex: 1 }} />
        <button onClick={buildForest} style={{ padding: "7px 14px", borderRadius: 6, border: `1px solid ${C.border}`, background: "#1a1f2e", color: C.text, fontSize: 11, fontFamily: "inherit", cursor: "pointer", fontWeight: 600 }}>↻ Rebuild</button>
        <button onClick={() => {
          if (growing) { cancelRef.current = true; growRef.current = false; setGrowing(false); }
          else autoGrow(curTree);
        }} style={{
          padding: "7px 18px", borderRadius: 6, border: "none",
          background: growing ? `linear-gradient(135deg,${C.red},#dc2626)` : `linear-gradient(135deg,${C.accent},${C.green})`,
          color: "#000", fontSize: 12, fontFamily: "inherit", cursor: "pointer", fontWeight: 800,
        }}>
          {growing ? "■ Stop" : "▶ Grow"}
        </button>
      </div>

      {/* Step controls */}
      <div style={{ display: "flex", alignItems: "center", gap: 6, padding: "7px 20px", borderBottom: `1px solid ${C.border}`, flexWrap: "wrap" }}>
        <button disabled={growing || ts.stepIdx <= -1} onClick={() => goToStep(curTree, ts.stepIdx - 1)}
          style={{ ...inp, cursor: growing || ts.stepIdx <= -1 ? "default" : "pointer", padding: "3px 10px", opacity: growing || ts.stepIdx <= -1 ? .3 : 1, fontSize: 11 }}>◀ Prev</button>
        <button disabled={growing || ts.stepIdx >= totalSteps - 1} onClick={() => goToStep(curTree, ts.stepIdx + 1)}
          style={{ ...inp, cursor: growing || ts.stepIdx >= totalSteps - 1 ? "default" : "pointer", padding: "3px 10px", opacity: growing || ts.stepIdx >= totalSteps - 1 ? .3 : 1, fontSize: 11 }}>Next ▶</button>
        <span style={{ fontSize: 10, color: C.dim }}>
          Step {Math.max(0, ts.stepIdx + 1)}/{totalSteps}{ts.stepIdx === -1 ? " (ready)" : ""}
        </span>
        {curBootstrap && (
          <span style={{ fontSize: 9, color: C.purple, marginLeft: 8 }}>
            Bootstrap: {curBootstrap.inBag}/{TOTAL_SAMPLES} in-bag · {curBootstrap.oob} OOB (acc={curBootstrap.oobAccuracy})
          </span>
        )}
        <div style={{ flex: 1 }} />
        <button onClick={() => setTS(curTree, EMPTY_TS)} disabled={growing}
          style={{ ...inp, cursor: "pointer", padding: "3px 10px", fontSize: 10, opacity: growing ? .3 : 1 }}>Reset</button>
      </div>

      {/* Tree tabs */}
      {nEstimators > 1 && (
        <div style={{ display: "flex", gap: 3, padding: "7px 20px", overflowX: "auto", borderBottom: `1px solid ${C.border}`, alignItems: "center" }}>
          {trees.map((_, i) => {
            const s = treeStates[i];
            const done = s && s.stepIdx >= getSteps(i).length - 1 && getSteps(i).length > 0;
            return (
              <button key={i}
                onClick={() => { if (!growing) setCurTree(i); }}
                onDoubleClick={() => { if (!growing) { setCurTree(i); instantComplete(i); } }}
                title="Click to view · Double-click to instantly complete"
                style={{
                  padding: "3px 10px", borderRadius: 4, border: "none", flexShrink: 0,
                  background: i === curTree ? C.accent : done ? C.green + "33" : "#1a1f2e",
                  color: i === curTree ? "#000" : done ? C.green : C.dim,
                  fontSize: 10, fontFamily: "inherit", cursor: growing ? "default" : "pointer",
                  fontWeight: i === curTree ? 700 : 400,
                }}>
                Tree {i + 1}{done ? " ✓" : ""}
              </button>
            );
          })}
          <div style={{ flex: 1 }} />
          <button onClick={growAllInstant} disabled={growing}
            title="Instantly complete all trees"
            style={{
              padding: "3px 12px", borderRadius: 4, border: `1px solid ${C.border}`,
              background: "#1a1f2e", color: C.accent, fontSize: 10,
              fontFamily: "inherit", cursor: growing ? "default" : "pointer",
              fontWeight: 600, opacity: growing ? .3 : 1, flexShrink: 0,
            }}>
            ⚡ Grow All
          </button>
        </div>
      )}

      {/* SVG Canvas */}
      <div ref={canvasRef}
        onMouseDown={onMouseDown} onMouseMove={onMouseMove}
        onMouseUp={onMouseUp} onMouseLeave={onMouseUp}
        style={{
          overflow: "hidden", cursor: isDragging.current ? "grabbing" : "grab",
          height: Math.min(500, svgH + 40), background: "#080c14",
          borderBottom: `1px solid ${C.border}`, position: "relative",
        }}>
        <svg width={treeWidth} height={svgH} viewBox={`0 0 ${treeWidth} ${svgH}`}
          style={{
            display: "block",
            transform: `translate(${pan.x}px,${pan.y}px) scale(${zoom})`,
            transformOrigin: "0 0",
          }}>
          <defs>
            <filter id="ng"><feDropShadow dx="0" dy="0" stdDeviation="3" floodColor={C.accent} floodOpacity=".25"/></filter>
            <filter id="lg"><feDropShadow dx="0" dy="0" stdDeviation="2.5" floodColor={C.blue} floodOpacity=".2"/></filter>
          </defs>
          {allNodes.filter(n => n.type === "split").map(n => (
            <g key={"e" + n.id}>
              <Edge p1={positions[n.id]} p2={positions[n.left?.id]}
                visible={visibleSet.has(n.id) && visibleSet.has(n.left?.id)} label="Yes" />
              <Edge p1={positions[n.id]} p2={positions[n.right?.id]}
                visible={visibleSet.has(n.id) && visibleSet.has(n.right?.id)} label="No" />
            </g>
          ))}
          {allNodes.map(n => (
            <TreeNode key={n.id} node={n} pos={positions[n.id]}
              show={visibleSet.has(n.id) || n.id === ts.nodeId}
              phase={n.id === ts.nodeId ? ts.phase : (visibleSet.has(n.id) ? 3 : 0)} />
          ))}
          {ts.nodeId && positions[ts.nodeId] && (
            <rect x={positions[ts.nodeId].x - 58} y={positions[ts.nodeId].y - 28}
              width={116} height={56} rx={10} fill="none"
              stroke={C.accent} strokeWidth={1} strokeDasharray="3 3" opacity={.5}>
              <animate attributeName="stroke-dashoffset" from="0" to="12" dur="1.5s" repeatCount="indefinite" />
            </rect>
          )}
        </svg>
      </div>

      {/* Feature panel — now shows ALL features with Gini */}
      <div style={{ padding: "8px 20px 4px" }}>
        <div style={{ fontSize: 9, color: C.dim, marginBottom: 4, textTransform: "uppercase", letterSpacing: 1 }}>
          Feature Pool — <span style={{ color: C.dimmer }}>●</span> not sampled{" "}
          <span style={{ color: C.orange }}>●</span> candidate subset{" "}
          <span style={{ color: C.green }}>●</span> best in subset
          {currentNode?.type === "split" && currentNode.globalBestIdx !== currentNode.featureIndex && ts.phase >= 2 && (
            <span style={{ color: C.red, marginLeft: 8 }}>⚠ true best was not in subset!</span>
          )}
        </div>
        <div style={{ display: "flex", flexWrap: "wrap", gap: 5, justifyContent: "center", padding: "10px 12px", background: "#0d1117", borderRadius: 8, border: `1px solid ${C.border}` }}>
          {FEATURES.map((f, i) => {
            const cn = currentNode?.type === "split" ? currentNode : null;
            const isCand = cn?.candidateIndices?.includes(i);
            const isBest = cn && i === cn.featureIndex;
            const isGlobalBest = cn && i === cn.globalBestIdx && cn.globalBestIdx !== cn.featureIndex;
            const ev = cn?.allFeatureEvals?.find(e => e.featureIndex === i);
            const showGini = ts.phase >= 1 && cn && ev;

            let bg = "#1a1f2e", col = C.dimmer, bdr = "1px solid transparent", shd = "none";
            if (ts.phase >= 2 && isBest) {
              bg = C.greenG; col = C.green; bdr = `1px solid ${C.green}`; shd = `0 0 10px ${C.greenG}`;
            } else if (ts.phase >= 1 && isCand) {
              bg = C.accentG; col = C.orange; bdr = `1px solid ${C.accent}55`; shd = `0 0 8px ${C.accentG}`;
            } else if (ts.phase >= 2 && isGlobalBest) {
              bg = "#1a1020"; col = C.red; bdr = `1px solid ${C.red}44`;
            }

            return (
              <div key={i} style={{
                padding: "4px 9px", borderRadius: 5, fontSize: 10, fontFamily: "inherit",
                fontWeight: (isBest && ts.phase >= 2) || (isGlobalBest && ts.phase >= 2) ? 700 : 400,
                background: bg, color: col, border: bdr, boxShadow: shd,
                transition: "all .35s", minWidth: 80, textAlign: "center",
              }}>
                {f}
                {showGini && (
                  <div style={{ fontSize: 7.5, marginTop: 1, opacity: isCand || (isGlobalBest && ts.phase >= 2) ? 1 : 0.5 }}>
                    G={ev.gini.toFixed(3)}
                    {ts.phase >= 2 && isGlobalBest && <span style={{ fontSize: 7, color: C.red }}> ← best</span>}
                  </div>
                )}
              </div>
            );
          })}
        </div>
      </div>

      {/* Calc panel */}
      <div style={{ padding: "4px 20px 8px" }}>
        <div style={{ fontSize: 9, color: C.dim, marginBottom: 4, textTransform: "uppercase", letterSpacing: 1 }}>Calculations</div>
        <div style={{ padding: 12, background: "#0d1117", borderRadius: 8, border: `1px solid ${C.border}`, fontSize: 10.5, lineHeight: 1.7 }}>
          {!currentNode && <span style={{ color: C.dim }}>Press ▶ Grow or → arrow key to begin…</span>}
          {currentNode?.type === "leaf" && (
            <span>🍃 <strong style={{ color: currentNode.prediction === "A" ? C.blue : C.leafB }}>Leaf</strong> — Predict: {currentNode.prediction} | Gini={currentNode.impurity.toFixed(4)} | n={currentNode.samples} [{currentNode.classA}A, {currentNode.classB}B]</span>
          )}
          {currentNode?.type === "split" && (<>
            <div style={{ color: C.accent, fontWeight: 700, marginBottom: 5, fontSize: 11 }}>▸ Node depth={currentNode.depth}, n={currentNode.samples}</div>
            {ts.phase >= 1 && (
              <div><span style={{ color: C.orange }}>① Random subset ({currentNode.candidateIndices.length}/{FEATURES.length}):</span> [{currentNode.candidateIndices.map(i => FEATURES[i]).join(", ")}]</div>
            )}
            {ts.phase >= 1 && (
              <div style={{ marginTop: 3 }}>
                <span style={{ color: C.orange }}>② Gini per candidate:</span>
                <div style={{ marginLeft: 14, marginTop: 2 }}>
                  {currentNode.allFeatureEvals
                    .filter(ev => currentNode.candidateIndices.includes(ev.featureIndex))
                    .map((ev, j) => (
                    <div key={j} style={{ color: ts.phase >= 2 && ev.featureIndex === currentNode.featureIndex ? C.green : C.dim }}>
                      {FEATURES[ev.featureIndex]}: t={ev.threshold} G={ev.gini.toFixed(4)}
                      {ts.phase >= 2 && ev.featureIndex === currentNode.featureIndex && " ◀ best in subset"}
                    </div>
                  ))}
                </div>
              </div>
            )}
            {ts.phase >= 2 && currentNode.globalBestIdx !== currentNode.featureIndex && (
              <div style={{ marginTop: 3, color: C.red, fontSize: 10 }}>
                ⚠ Global best was <strong>{FEATURES[currentNode.globalBestIdx]}</strong> (G={currentNode.globalBestGini.toFixed(4)}) but it wasn't in the random subset
              </div>
            )}
            {ts.phase >= 2 && (
              <div style={{ marginTop: 3 }}><span style={{ color: C.green }}>③ Split:</span> <strong>{currentNode.featureName}</strong> ≤ {currentNode.threshold} (G={currentNode.gini.toFixed(4)})</div>
            )}
          </>)}
        </div>
      </div>

      {/* Ensemble Vote Panel */}
      {nEstimators > 1 && (
        <div style={{ padding: "0 20px 14px" }}>
          <div style={{ fontSize: 9, color: C.dim, marginBottom: 4, textTransform: "uppercase", letterSpacing: 1 }}>
            Ensemble — Majority Vote
          </div>
          <div style={{ padding: 14, background: "#0d1117", borderRadius: 8, border: `1px solid ${C.border}` }}>
            {completedTrees.length === 0 ? (
              <span style={{ fontSize: 10.5, color: C.dim }}>Grow trees to see the ensemble prediction. Each tree votes, majority wins.</span>
            ) : (
              <div>
                {/* Vote visualization */}
                <div style={{ display: "flex", gap: 6, marginBottom: 10, flexWrap: "wrap", alignItems: "center" }}>
                  {trees.map((_, i) => {
                    const ct = completedTrees.find(t => t.idx === i);
                    return (
                      <div key={i} style={{
                        width: 38, height: 38, borderRadius: 6, display: "flex",
                        flexDirection: "column", alignItems: "center", justifyContent: "center",
                        background: ct ? (ct.prediction === "A" ? C.blue + "22" : C.leafB + "22") : "#151a24",
                        border: `1px solid ${ct ? (ct.prediction === "A" ? C.blue + "66" : C.leafB + "66") : C.border}`,
                        transition: "all .3s",
                      }}>
                        <div style={{ fontSize: 7, color: C.dim }}>T{i + 1}</div>
                        <div style={{
                          fontSize: 11, fontWeight: 700,
                          color: ct ? (ct.prediction === "A" ? C.blue : C.leafB) : C.dimmer,
                        }}>
                          {ct ? ct.prediction : "—"}
                        </div>
                      </div>
                    );
                  })}
                  <div style={{ marginLeft: 8, fontSize: 10, color: C.dim }}>→</div>
                  {hasEnsemble && (
                    <div style={{
                      padding: "6px 14px", borderRadius: 6,
                      background: ensemblePrediction === "A"
                        ? `linear-gradient(135deg,${C.blue}33,${C.blue}11)`
                        : `linear-gradient(135deg,${C.leafB}33,${C.leafB}11)`,
                      border: `2px solid ${ensemblePrediction === "A" ? C.blue : C.leafB}`,
                    }}>
                      <div style={{ fontSize: 8, color: C.dim, textTransform: "uppercase" }}>Prediction</div>
                      <div style={{
                        fontSize: 16, fontWeight: 800,
                        color: ensemblePrediction === "A" ? C.blue : C.leafB,
                      }}>
                        Class {ensemblePrediction}
                      </div>
                    </div>
                  )}
                </div>

                {/* Tally */}
                <div style={{ display: "flex", gap: 16, fontSize: 10, color: C.dim }}>
                  <span>Votes: <strong style={{ color: C.blue }}>{votesA}× A</strong> vs <strong style={{ color: C.leafB }}>{votesB}× B</strong></span>
                  <span>Completed: {completedTrees.length}/{nEstimators}</span>
                  {bootstrapInfo.length > 0 && (
                    <span style={{ color: C.purple }}>
                      Avg OOB acc: {(bootstrapInfo.reduce((s, b) => s + b.oobAccuracy, 0) / bootstrapInfo.length).toFixed(3)}
                    </span>
                  )}
                </div>

                {/* Vote bar */}
                {hasEnsemble && (
                  <div style={{ marginTop: 8, display: "flex", height: 8, borderRadius: 4, overflow: "hidden", background: "#151a24" }}>
                    <div style={{ width: `${(votesA / (votesA + votesB)) * 100}%`, background: C.blue, transition: "width .3s" }} />
                    <div style={{ width: `${(votesB / (votesA + votesB)) * 100}%`, background: C.leafB, transition: "width .3s" }} />
                  </div>
                )}
              </div>
            )}
          </div>
        </div>
      )}

      {/* Footer */}
      <div style={{ padding: "8px 20px", borderTop: `1px solid ${C.border}`, fontSize: 8.5, color: C.dim, textAlign: "center" }}>
        Gini = 1 − Σpᵢ² | Weighted = (nₗ/n)Gₗ + (nᵣ/n)Gᵣ | max_features={subsetSize}/{FEATURES.length} | Trees: {nEstimators} | Bootstrap: n={TOTAL_SAMPLES}
      </div>
    </div>
  );
}
