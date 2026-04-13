import { useState, useEffect, useRef, useCallback } from "react";
import { useLocation } from "react-router-dom";
import Papa from "papaparse";
import { C } from "./theme";
import GlobalHeader from "./GlobalHeader";
import { heartData, heartMeta } from "./data/heartDisease";
import { buildRealTree, bootstrapSample, computeOOBAccuracy } from "./cartAlgorithm";

// ─── Feature-subset options ────────────────────────────────────────────────────
const FEATURE_SUBSET_OPTIONS = {
  sqrt: { label: "√p (sqrt)", fn: (p) => Math.max(1, Math.round(Math.sqrt(p))) },
  log2: { label: "log₂(p)",  fn: (p) => Math.max(1, Math.round(Math.log2(p))) },
  half: { label: "p/2",       fn: (p) => Math.max(1, Math.round(p / 2)) },
  all:  { label: "p (all)",   fn: (p) => p },
};

// ─── Tree helpers ──────────────────────────────────────────────────────────────
function flattenNodes(node, id = "0") {
  node.id = id;
  const nodes = [node];
  if (node.type === "split") {
    nodes.push(...flattenNodes(node.left,  id + "L"));
    nodes.push(...flattenNodes(node.right, id + "R"));
  }
  return nodes;
}

function getTreePrediction(node) {
  const leaves = flattenNodes(node).filter(n => n.type === "leaf");
  let totalA = 0, totalB = 0;
  leaves.forEach(l => { totalA += l.classA; totalB += l.classB; });
  return totalA >= totalB ? "A" : "B";
}

// Count actual leaf nodes so width scales with real tree shape, not worst-case depth.
function countLeaves(node) {
  if (!node || node.type === "leaf") return 1;
  return countLeaves(node.left) + countLeaves(node.right);
}

const LEAF_SPACING = 130; // px between leaf centres
const X_PAD = 30;         // left/right margin
const Y_GAP = 90;         // vertical distance between depth levels

function computeTreeWidth(node) {
  return X_PAD * 2 + countLeaves(node) * LEAF_SPACING;
}

// Positions each node so horizontal space is proportional to its subtree's leaf count.
// This prevents exponential width blow-up at deep depths.
function computePositions(node) {
  const pos = {};
  function layout(n, leafOffset, depth) {
    const leaves = countLeaves(n);
    pos[n.id] = { x: X_PAD + (leafOffset + leaves / 2) * LEAF_SPACING, y: 40 + depth * Y_GAP };
    if (n.type === "split") {
      const lLeaves = countLeaves(n.left);
      layout(n.left,  leafOffset,           depth + 1);
      layout(n.right, leafOffset + lLeaves, depth + 1);
    }
  }
  layout(node, 0, 0);
  return pos;
}

// Split a feature name into up to 2 lines for SVG rendering.
// Returns { line1, line2 } where line2 may be null.
function splitFeatureName(name, maxChars = 22) {
  if (name.length <= maxChars) return { line1: name, line2: null };
  // Try to break at a word boundary near the midpoint
  const mid = Math.floor(name.length / 2);
  let breakAt = -1;
  for (let d = 0; d <= mid; d++) {
    if (name[mid - d] === " ") { breakAt = mid - d; break; }
    if (name[mid + d] === " ") { breakAt = mid + d; break; }
  }
  if (breakAt === -1) {
    // No space found — hard-split at maxChars
    let l1 = name.slice(0, maxChars);
    let l2 = name.slice(maxChars);
    if (l2.length > maxChars) l2 = l2.slice(0, maxChars - 1) + "…";
    return { line1: l1, line2: l2 };
  }
  let line1 = name.slice(0, breakAt);
  let line2 = name.slice(breakAt + 1);
  if (line1.length > maxChars) line1 = line1.slice(0, maxChars - 1) + "…";
  if (line2.length > maxChars) line2 = line2.slice(0, maxChars - 1) + "…";
  return { line1, line2 };
}

// ─── SVG components ────────────────────────────────────────────────────────────
function Edge({ p1, p2, visible, label }) {
  if (!visible || !p1 || !p2) return null;
  return (
    <g style={{ opacity: 1, transition: "opacity .3s" }}>
      <line x1={p1.x} y1={p1.y + 27} x2={p2.x} y2={p2.y - 24}
        stroke={C.edge} strokeWidth={1.4} strokeDasharray="4 2" />
      <text x={(p1.x + p2.x) / 2} y={(p1.y + 27 + p2.y - 24) / 2 - 4}
        textAnchor="middle" fill={C.dim} fontSize={8} fontFamily="'JetBrains Mono',monospace">{label}</text>
    </g>
  );
}

function TreeNode({ node, show, phase, pos, labelA, labelB }) {
  if (!show || !pos) return null;
  const { x, y } = pos;

  // ── Leaf node — unchanged behaviour (invisible at phase 0, full reveal at phase 1) ──
  if (node.type === "leaf") {
    const vis = phase >= 1 ? 1 : 0;
    const r = node.samples > 0 ? node.classA / node.samples : 0.5;
    const leafLabel = node.prediction === "A" ? labelA : labelB;
    const displayLabel = leafLabel.length > 11 ? leafLabel.slice(0, 10) + "…" : leafLabel;
    return (
      <g style={{ opacity: vis, transition: "opacity .5s" }}>
        <rect x={x - 38} y={y - 24} width={76} height={50} rx={7}
          fill={C.panel} stroke={node.prediction === "A" ? C.blue : C.leafB} strokeWidth={1.8} filter="url(#lg)" />
        <text x={x} y={y - 9} textAnchor="middle" fill={C.text} fontSize={9}
          fontFamily="'JetBrains Mono',monospace" fontWeight={600}>
          {displayLabel}</text>
        <rect x={x - 28} y={y + 2}  width={56}                     height={5} rx={2.5} fill={C.leafB} opacity={0.35} />
        <rect x={x - 28} y={y + 2}  width={Math.max(0, 56 * r)}    height={5} rx={2.5} fill={C.blue} />
        <text x={x} y={y + 17} textAnchor="middle" fill={C.dim} fontSize={7.5}
          fontFamily="'JetBrains Mono',monospace">n={node.samples} G={node.impurity.toFixed(3)}</text>
      </g>
    );
  }

  // ── Split node ─────────────────────────────────────────────────────────────
  // phase 0/1: node visible but feature unknown — show "?" placeholder + sample count
  // phase 2+:  feature revealed — show full label, threshold, Gini
  const revealed = phase >= 2;
  const borderColor  = revealed ? C.accent  : C.orange;
  const filterId     = revealed ? "url(#ng)" : "url(#ng-p)";

  const { line1, line2 } = splitFeatureName(node.featureName ?? "");

  return (
    <g style={{ transition: "opacity .4s" }}>
      {/* Box — wider (180px) and taller (54px) to fit long feature names */}
      <rect x={x - 90} y={y - 27} width={180} height={54} rx={5}
        fill={C.panel}
        stroke={borderColor}
        strokeWidth={revealed ? 1.4 : 1.8}
        filter={filterId}
        style={{ transition: "stroke 0.25s, stroke-width 0.25s" }}
      />

      {/* Placeholder — "?" + sample count, fades out on reveal */}
      <g style={{ opacity: revealed ? 0 : 1, transition: "opacity 0.2s" }}>
        <text x={x} y={y - 5} textAnchor="middle" fill={C.orange} fontSize={11}
          fontFamily="'JetBrains Mono',monospace" fontWeight={700}>?</text>
        <text x={x} y={y + 11} textAnchor="middle" fill={C.dim} fontSize={7.5}
          fontFamily="'JetBrains Mono',monospace">n={node.samples}</text>
      </g>

      {/* Revealed label — 1 or 2 line feature name + threshold + Gini */}
      <g style={{ opacity: revealed ? 1 : 0, transition: "opacity 0.3s" }}>
        {line2 ? (
          <>
            <text x={x} y={y - 13} textAnchor="middle" fill={C.accent} fontSize={8.5}
              fontFamily="'JetBrains Mono',monospace" fontWeight={700}>{line1}</text>
            <text x={x} y={y - 2} textAnchor="middle" fill={C.accent} fontSize={8.5}
              fontFamily="'JetBrains Mono',monospace" fontWeight={700}>{line2}</text>
            <text x={x} y={y + 14} textAnchor="middle" fill={C.dim} fontSize={7.5}
              fontFamily="'JetBrains Mono',monospace">≤{node.threshold} G={node.gini.toFixed(3)} n={node.samples}</text>
          </>
        ) : (
          <>
            <text x={x} y={y - 6} textAnchor="middle" fill={C.accent} fontSize={9}
              fontFamily="'JetBrains Mono',monospace" fontWeight={700}>{line1}</text>
            <text x={x} y={y + 10} textAnchor="middle" fill={C.dim} fontSize={7.5}
              fontFamily="'JetBrains Mono',monospace">≤{node.threshold} G={node.gini.toFixed(3)} n={node.samples}</text>
          </>
        )}
      </g>
    </g>
  );
}

const EMPTY_TS = { visibleIds: [], nodeId: null, phase: 0, stepIdx: -1 };

// ─── Interaction hint tooltip (shown via ? button in header) ──────────────────
function HintTooltip() {
  const [pos, setPos] = useState(null);
  const btnRef = useRef(null);
  const show = () => {
    const r = btnRef.current?.getBoundingClientRect();
    if (r) setPos({ left: r.left + r.width / 2, top: r.bottom + 6 });
  };
  return (
    <span style={{ display: "inline-block", marginLeft: 5, verticalAlign: "middle" }}>
      <span
        ref={btnRef}
        onMouseEnter={show}
        onMouseLeave={() => setPos(null)}
        style={{
          cursor: "help", fontSize: 9, color: C.dimmer,
          border: `1px solid ${C.dimmer}`, borderRadius: "50%",
          width: 14, height: 14, display: "inline-flex",
          alignItems: "center", justifyContent: "center",
          lineHeight: 1, userSelect: "none",
        }}
      >?</span>
      {pos && (
        <div style={{
          position: "fixed", left: pos.left, top: pos.top,
          transform: "translateX(-50%)", zIndex: 1000,
          background: "#1a2035", border: `1px solid ${C.border}`,
          borderRadius: 8, padding: "10px 14px",
          fontSize: 10, color: C.dim, lineHeight: 1.8,
          whiteSpace: "nowrap", pointerEvents: "none",
          boxShadow: "0 4px 20px rgba(0,0,0,0.55)",
        }}>
          <div>Scroll / pinch to <strong style={{ color: C.text }}>zoom</strong></div>
          <div>Drag to <strong style={{ color: C.text }}>pan</strong></div>
          <div><strong style={{ color: C.text }}>← →</strong> arrow keys to step</div>
          <div>Double-click a tab to <strong style={{ color: C.text }}>instantly complete</strong></div>
        </div>
      )}
    </span>
  );
}

// ─── CSV processing helpers ────────────────────────────────────────────────────
const NA_VALS = new Set(["", "NA", "na", "nan", "NaN", "?", "null", "undefined", "N/A", "n/a"]);

// Turn a raw target value into a display label.
// Numeric values get a "Class " prefix; string values are used as-is (capitalised).
function formatClassLabel(raw) {
  const s = String(raw).trim();
  if (s === "") return "Unknown";
  const n = Number(s);
  if (!isNaN(n) && s !== "") return `Class ${s}`;
  return s.charAt(0).toUpperCase() + s.slice(1);
}

function detectNAs(rows, headers) {
  let total = 0;
  const byCol = {};
  headers.forEach(h => { byCol[h] = 0; });
  rows.forEach(r => headers.forEach(h => {
    if (NA_VALS.has(String(r[h] ?? "").trim())) { total++; byCol[h]++; }
  }));
  return { total, byCol };
}

// Stratified random sample without replacement, preserving class proportions.
function stratifiedSample(rows, targetCol, n) {
  // Group by raw target value
  const groups = new Map();
  rows.forEach(r => {
    const key = String(r[targetCol] ?? "");
    if (!groups.has(key)) groups.set(key, []);
    groups.get(key).push(r);
  });

  const total = rows.length;
  const result = [];
  for (const [, group] of groups) {
    const take = Math.max(1, Math.round((group.length / total) * n));
    // Fisher-Yates shuffle in-place on a copy, then slice
    const g = [...group];
    for (let i = g.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [g[i], g[j]] = [g[j], g[i]];
    }
    result.push(...g.slice(0, Math.min(take, g.length)));
  }
  // Final shuffle so classes are interleaved
  for (let i = result.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [result[i], result[j]] = [result[j], result[i]];
  }
  return result;
}

function processCSVData(rawRows, headers, targetCol, naStrategy, sampleMode) {
  // Apply NA strategy
  let rows;
  if (naStrategy === "drop") {
    rows = rawRows.filter(r => headers.every(h => !NA_VALS.has(String(r[h] ?? "").trim())));
  } else {
    // Fill non-target numeric columns with column median
    const medians = {};
    headers.filter(h => h !== targetCol).forEach(h => {
      const nums = rawRows.map(r => parseFloat(r[h])).filter(v => !isNaN(v)).sort((a, b) => a - b);
      if (nums.length) medians[h] = nums[Math.floor(nums.length / 2)];
    });
    rows = rawRows.map(r => {
      const copy = { ...r };
      headers.forEach(h => {
        if (h !== targetCol && NA_VALS.has(String(r[h] ?? "").trim())) {
          copy[h] = String(medians[h] ?? 0);
        }
      });
      return copy;
    });
  }

  if (rows.length === 0) return null;

  // Apply stratified sampling before encoding so class proportions are preserved.
  const totalRows = rows.length;
  if (sampleMode === "1000" && rows.length > 1000) {
    rows = stratifiedSample(rows, targetCol, 1000);
  } else if (sampleMode === "2000" && rows.length > 2000) {
    rows = stratifiedSample(rows, targetCol, 2000);
  }
  const sampledRows = rows.length;

  const featCols = headers.filter(h => h !== targetCol);

  // Determine numeric vs categorical
  const isNum = {};
  featCols.forEach(h => {
    const vals = rows.map(r => String(r[h] ?? "").trim()).filter(v => !NA_VALS.has(v));
    isNum[h] = vals.length > 0 && vals.every(v => v !== "" && !isNaN(parseFloat(v)));
  });

  // Unique values for categoricals (for one-hot)
  const catVals = {};
  featCols.filter(h => !isNum[h]).forEach(h => {
    catVals[h] = [...new Set(rows.map(r => String(r[h] ?? "")))].sort();
  });

  // Build feature name list
  const features = [];
  featCols.forEach(h => {
    if (isNum[h]) {
      features.push(h);
    } else {
      // one-hot, drop first category
      catVals[h].slice(1).forEach(v => features.push(`${h}_${v}`));
    }
  });

  // Target binarisation: first sorted unique value → 0 (class A), rest → 1 (class B).
  // Capture raw values before binarisation so we can build human-readable labels.
  const targetUniq = [...new Set(rows.map(r => String(r[targetCol] ?? "")))].sort();
  const classLabels = {
    A: formatClassLabel(targetUniq[0] ?? "A"),
    B: targetUniq.length === 2 ? formatClassLabel(targetUniq[1]) : "Other",
  };

  const data = rows.map(r => {
    const obj = {};
    featCols.forEach(h => {
      if (isNum[h]) {
        obj[h] = parseFloat(r[h]) || 0;
      } else {
        catVals[h].slice(1).forEach(v => {
          obj[`${h}_${v}`] = String(r[h] ?? "") === v ? 1 : 0;
        });
      }
    });
    obj["target"] = String(r[targetCol] ?? "") === targetUniq[0] ? 0 : 1;
    return obj;
  });

  return { data, features, targetCol: "target", totalRows, sampledRows, classLabels };
}

const SAMPLE_OPTIONS = [
  { key: "all",  label: "Use all rows",          sub: "May be slow for large datasets" },
  { key: "2000", label: "Random sample: 2,000",  sub: "Stratified by target class" },
  { key: "1000", label: "Random sample: 1,000",  sub: "Stratified by target class" },
];

// ─── CSV Modal ─────────────────────────────────────────────────────────────────
function DataModal({ modal, onUpdate, onConfirm, onCancel }) {
  const { fileName, rawRows, headers, naStats, selectedTarget, naStrategy, sampleMode, warning } = modal;
  const isLarge = rawRows.length > 2000;
  const inp = {
    padding: "5px 8px", borderRadius: 5, background: "#1a1f2e",
    border: `1px solid ${C.border}`, color: C.text, fontSize: 11,
    fontFamily: "'JetBrains Mono',monospace", outline: "none",
  };

  const handleConfirm = () => {
    const result = processCSVData(rawRows, headers, selectedTarget, naStrategy, sampleMode ?? "all");
    if (!result) return;
    onConfirm(result.data, result.features, result.targetCol,
              fileName.replace(".csv", ""), result.totalRows, result.sampledRows, result.classLabels);
  };

  return (
    <div style={{
      position: "fixed", inset: 0, background: "rgba(0,0,0,0.75)", zIndex: 100,
      display: "flex", alignItems: "center", justifyContent: "center",
    }}>
      <div style={{
        background: C.panel, border: `1px solid ${C.border}`, borderRadius: 12,
        padding: 24, maxWidth: 480, width: "90vw", fontFamily: "'JetBrains Mono',monospace",
        color: C.text, maxHeight: "85vh", overflowY: "auto",
      }}>
        <div style={{ fontSize: 13, fontWeight: 700, color: C.accent, marginBottom: 14 }}>
          Configure Dataset
        </div>

        {/* File summary */}
        <div style={{ fontSize: 10, color: C.dim, marginBottom: 12, lineHeight: 1.8 }}>
          <div>📄 <strong style={{ color: C.text }}>{fileName}</strong></div>
          <div>{rawRows.length.toLocaleString()} rows · {headers.length} columns</div>
          {naStats.total > 0 && (
            <div style={{ color: C.orange }}>⚠ {naStats.total} missing values found</div>
          )}
          {warning && <div style={{ color: C.red, marginTop: 4 }}>⚠ {warning}</div>}
        </div>

        {/* Target column */}
        <div style={{ marginBottom: 14 }}>
          <label style={{ fontSize: 9, color: C.dim, textTransform: "uppercase", letterSpacing: 1, display: "block", marginBottom: 4 }}>
            Target column (class to predict)
          </label>
          <select value={selectedTarget} onChange={e => onUpdate({ selectedTarget: e.target.value })} style={{ ...inp, width: "100%", cursor: "pointer" }}>
            {headers.map(h => <option key={h} value={h}>{h}</option>)}
          </select>
          <div style={{ fontSize: 9, color: C.dimmer, marginTop: 3 }}>
            First unique value → Class A, all others → Class B
          </div>
        </div>

        {/* NA strategy */}
        {naStats.total > 0 && (
          <div style={{ marginBottom: 14 }}>
            <label style={{ fontSize: 9, color: C.dim, textTransform: "uppercase", letterSpacing: 1, display: "block", marginBottom: 6 }}>
              Handle missing values
            </label>
            <div style={{ display: "flex", gap: 8 }}>
              {["drop", "median"].map(s => (
                <button key={s} onClick={() => onUpdate({ naStrategy: s })} style={{
                  flex: 1, padding: "6px 10px", borderRadius: 6, fontSize: 10, cursor: "pointer",
                  fontFamily: "inherit", fontWeight: naStrategy === s ? 700 : 400,
                  background: naStrategy === s ? C.accentG : "#1a1f2e",
                  border: `1px solid ${naStrategy === s ? C.accent : C.border}`,
                  color: naStrategy === s ? C.accent : C.dim,
                }}>
                  {s === "drop" ? "Drop rows with NA" : "Fill with median"}
                </button>
              ))}
            </div>
          </div>
        )}

        {/* Sampling — only shown for large datasets */}
        {isLarge && (
          <div style={{ marginBottom: 14 }}>
            <div style={{
              padding: "8px 11px", borderRadius: 6, background: `${C.orange}12`,
              border: `1px solid ${C.orange}44`, fontSize: 10, color: C.orange,
              marginBottom: 10, lineHeight: 1.6,
            }}>
              Large dataset detected ({rawRows.length.toLocaleString()} rows). For smooth
              visualization, we recommend sampling.
            </div>
            <label style={{ fontSize: 9, color: C.dim, textTransform: "uppercase", letterSpacing: 1, display: "block", marginBottom: 6 }}>
              Sampling
            </label>
            <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
              {SAMPLE_OPTIONS.map(opt => {
                const active = (sampleMode ?? "2000") === opt.key;
                return (
                  <button key={opt.key} onClick={() => onUpdate({ sampleMode: opt.key })} style={{
                    padding: "7px 11px", borderRadius: 6, fontSize: 10, cursor: "pointer",
                    fontFamily: "inherit", textAlign: "left", display: "flex", justifyContent: "space-between", alignItems: "center",
                    background: active ? C.accentG : "#1a1f2e",
                    border: `1px solid ${active ? C.accent : C.border}`,
                    color: active ? C.accent : C.dim,
                    fontWeight: active ? 700 : 400,
                  }}>
                    <span>{opt.label}</span>
                    <span style={{ fontSize: 8, opacity: 0.7 }}>{opt.sub}</span>
                  </button>
                );
              })}
            </div>
          </div>
        )}

        {/* Privacy note */}
        <div style={{ fontSize: 9, color: C.dimmer, marginBottom: 16, textAlign: "center" }}>
          🔒 Your data never leaves your browser
        </div>

        {/* Actions */}
        <div style={{ display: "flex", gap: 8 }}>
          <button onClick={onCancel} style={{
            flex: 1, padding: "8px", borderRadius: 6, border: `1px solid ${C.border}`,
            background: "#1a1f2e", color: C.dim, fontSize: 11, fontFamily: "inherit", cursor: "pointer",
          }}>Cancel</button>
          <button onClick={handleConfirm} style={{
            flex: 2, padding: "8px", borderRadius: 6, border: "none",
            background: `linear-gradient(135deg,${C.accent},${C.green})`,
            color: "#000", fontSize: 11, fontFamily: "inherit", cursor: "pointer", fontWeight: 700,
          }}>Confirm & Build Forest</button>
        </div>
      </div>
    </div>
  );
}

// ─── Main component ────────────────────────────────────────────────────────────
export default function RandomForestViz({ mode = "random-forest" }) {
  const lockedNEstimators = mode === "decision-tree" ? 1 : null;
  const lockedMaxFeatures = (mode === "decision-tree" || mode === "bagging") ? "all" : null;

  // ── Dataset state ──────────────────────────────────────────────────────────
  const [customDataset, setCustomDataset] = useState(null);
  const [csvModal, setCsvModal]           = useState(null);
  const [dragOver, setDragOver]           = useState(false);
  const [badgeHovered, setBadgeHovered]   = useState(false);

  const activeData      = customDataset?.data        ?? heartData;
  const activeFeatures  = customDataset?.features    ?? heartMeta.features;
  const activeTargetCol = customDataset?.targetCol   ?? heartMeta.targetCol;
  const classLabels     = customDataset?.classLabels ?? heartMeta.targetLabels;
  const datasetLabel   = customDataset?.name
    ? (() => {
        const { name, totalRows, sampledRows } = customDataset;
        const feats = activeFeatures.length;
        if (totalRows && sampledRows < totalRows) {
          return `${name} — ${sampledRows.toLocaleString()}/${totalRows.toLocaleString()} sampled rows, ${feats} features`;
        }
        return `${name} — ${activeData.length.toLocaleString()} rows, ${feats} features`;
      })()
    : heartMeta.description;
  const datasetTooltip = customDataset ? null : heartMeta.tooltip;

  // ── Hyperparameter state ───────────────────────────────────────────────────
  const [maxDepth, setMaxDepth]         = useState(3);
  const [maxDepthStr, setMaxDepthStr]   = useState("3");
  const [featureSubset, setFeatureSubset] = useState(lockedMaxFeatures ?? "sqrt");
  const [nEstimators, setNEstimators]   = useState(lockedNEstimators ?? 3);
  const [nEstimatorsStr, setNEstimatorsStr] = useState(String(lockedNEstimators ?? 3));
  const [trees, setTrees]               = useState([]);
  const [bootstrapInfo, setBootstrapInfo] = useState([]);
  const [curTree, setCurTree]           = useState(0);
  const [treeStates, setTreeStates]     = useState({});
  const [growing, setGrowing]           = useState(false);
  const [speed, setSpeed]               = useState(1);
  const location = useLocation();
  const growRef   = useRef(false);
  const cancelRef = useRef(false);

  const [zoom, setZoom]                     = useState(1);
  const [pan, setPan]                       = useState({ x: 0, y: 0 });
  const [cursorGrabbing, setCursorGrabbing] = useState(false);
  const isDragging = useRef(false);
  const dragStart  = useRef({ x: 0, y: 0, px: 0, py: 0 });
  const canvasRef  = useRef(null);

  const subsetSize = FEATURE_SUBSET_OPTIONS[featureSubset].fn(activeFeatures.length);

  const getTS = (idx) => treeStates[idx] || EMPTY_TS;

  const setTS = useCallback((idx, patch) => {
    setTreeStates(prev => {
      const old = prev[idx] || EMPTY_TS;
      return { ...prev, [idx]: { ...old, ...patch } };
    });
  }, []);

  // ── Forest building ────────────────────────────────────────────────────────
  // Accepts data explicitly so it can be called right after state updates.
  const buildForestWithData = useCallback((data, features, targetCol) => {
    cancelRef.current = true;
    growRef.current   = false;
    setGrowing(false);
    const n       = data.length;
    const activeS = lockedMaxFeatures ?? featureSubset;
    const subSize = FEATURE_SUBSET_OPTIONS[activeS].fn(features.length);
    const f = [], bInfo = [];
    for (let i = 0; i < nEstimators; i++) {
      const { bootstrapData, oobIndices, inBag } = bootstrapSample(data);
      const t  = buildRealTree(bootstrapData, features, targetCol, maxDepth, subSize);
      flattenNodes(t);
      const oobAcc = computeOOBAccuracy(t, data, oobIndices, features, targetCol);
      bInfo.push({ inBag, oob: n - inBag, oobAccuracy: oobAcc ?? 0, bootstrapN: n });
      f.push(t);
    }
    setTrees(f);
    setBootstrapInfo(bInfo);
    setCurTree(0);
    setTreeStates({});
    setZoom(1);
    // pan is reset by the centerTree effect once trees state updates
  }, [maxDepth, featureSubset, nEstimators, lockedMaxFeatures]);

  const buildForest = useCallback(() => {
    buildForestWithData(activeData, activeFeatures, activeTargetCol);
  }, [buildForestWithData, activeData, activeFeatures, activeTargetCol]);

  useEffect(() => {
    const pending = location.state?.pendingCSV;
    if (pending) {
      // A CSV was dropped on the landing page — open the DataModal automatically
      const { data: rawRows, meta } = Papa.parse(pending.content, { header: true, skipEmptyLines: true });
      if (rawRows.length) {
        const headers = meta.fields;
        const naStats = detectNAs(rawRows, headers);
        const isLarge = rawRows.length > 2000;
        const modal = {
          fileName: pending.name, rawRows, headers, naStats,
          selectedTarget: headers[headers.length - 1],
          naStrategy: "drop", sampleMode: isLarge ? "2000" : "all",
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

  const autoGrow = useCallback(async (treeIdx) => {
    if (growRef.current) return;
    growRef.current   = true;
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

  // ── Keyboard handler ───────────────────────────────────────────────────────
  useEffect(() => {
    const h = (e) => {
      if (growing) return;
      if (e.key === "ArrowRight" || e.key === "ArrowDown") {
        e.preventDefault();
        goToStep(curTree, getTS(curTree).stepIdx + 1);
      } else if (e.key === "ArrowLeft" || e.key === "ArrowUp") {
        e.preventDefault();
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
      const factor = 1 + delta;
      setZoom(z => {
        const nz = Math.max(0.08, Math.min(4, z * factor));
        const sc = nz / z;
        setPan(p => ({ x: mx - sc * (mx - p.x), y: my - sc * (my - p.y) }));
        return nz;
      });
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

  useEffect(() => {
    centerTree(curTree);
  }, [curTree, trees, centerTree]);

  // ── File drop handling ─────────────────────────────────────────────────────
  const openFile = useCallback((file) => {
    if (!file || !file.name.toLowerCase().endsWith(".csv")) return;
    const reader = new FileReader();
    reader.onload = (e) => {
      const { data: rawRows, meta } = Papa.parse(e.target.result, { header: true, skipEmptyLines: true });
      if (!rawRows.length) return;
      const headers    = meta.fields;
      const naStats    = detectNAs(rawRows, headers);
      const isLarge    = rawRows.length > 2000;
      const sampleMode = isLarge ? "2000" : "all";
      setCsvModal({
        fileName: file.name, rawRows, headers, naStats,
        selectedTarget: headers[headers.length - 1],
        naStrategy: "drop", sampleMode,
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
  const handleDataConfirm = useCallback((data, features, targetCol, name, totalRows, sampledRows, classLabels) => {
    setCustomDataset({ data, features, targetCol, name, totalRows, sampledRows, classLabels });
    setCsvModal(null);
    buildForestWithData(data, features, targetCol);
  }, [buildForestWithData]);

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
    const s = treeStates[i];
    const steps = getSteps(i);
    const done  = s && s.stepIdx >= steps.length - 1 && steps.length > 0;
    return done ? { idx: i, prediction: getTreePrediction(t) } : null;
  }).filter(Boolean);

  const votesA = completedTrees.filter(t => t.prediction === "A").length;
  const votesB = completedTrees.filter(t => t.prediction === "B").length;
  const ensemblePrediction = votesA >= votesB ? "A" : "B";
  const hasEnsemble = completedTrees.length >= 2;

  const curBootstrap  = bootstrapInfo[curTree];
  const TOTAL_SAMPLES = activeData.length;

  const inp = {
    padding: "6px 8px", borderRadius: 6, background: "#1a1f2e",
    border: `1px solid ${C.border}`, color: C.text, fontSize: 12,
    fontFamily: "'JetBrains Mono',monospace", outline: "none",
  };

  // ── Render ─────────────────────────────────────────────────────────────────
  return (
    <div
      style={{ minHeight: "100vh", background: C.bg, color: C.text, fontFamily: "'JetBrains Mono','Fira Code',monospace" }}
      onDragOver={e => { e.preventDefault(); setDragOver(true); }}
      onDragLeave={() => setDragOver(false)}
      onDrop={handleDrop}
    >
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

      {/* Global Header */}
      <GlobalHeader
        breadcrumb={[
          { label: "Trees", route: "/" },
          { label: mode === "decision-tree" ? "Decision Tree" : mode === "bagging" ? "Bagging" : "Random Forest" },
        ]}
        description={
          <span>
            {mode === "decision-tree"
              ? "no ensemble · no feature subsampling"
              : mode === "bagging"
              ? "bootstrap ensemble · all features at each split"
              : "bootstrap ensemble · random feature subsets"}
            <HintTooltip />
          </span>
        }
        center={
          <span
            style={{ position: "relative", display: "inline-block" }}
            onMouseEnter={() => setBadgeHovered(true)}
            onMouseLeave={() => setBadgeHovered(false)}
          >
            <span style={{
              fontSize: 9, color: C.purple, background: `${C.purple}18`,
              border: `1px solid ${C.purple}44`, borderRadius: 4, padding: "2px 7px",
              cursor: datasetTooltip ? "help" : "default",
              userSelect: "none",
            }}>
              {datasetLabel}{datasetTooltip ? " ⓘ" : ""}
            </span>
            {datasetTooltip && badgeHovered && (
              <div style={{
                position: "absolute", top: "calc(100% + 6px)", left: "50%",
                transform: "translateX(-50%)", zIndex: 20,
                background: "#1a2035", border: `1px solid ${C.purple}66`,
                borderRadius: 7, padding: "10px 13px", width: 300,
                fontSize: 10, color: C.text, lineHeight: 1.6,
                boxShadow: `0 4px 20px rgba(0,0,0,0.5)`,
                pointerEvents: "none",
              }}>
                {datasetTooltip}
              </div>
            )}
          </span>
        }
        right={
          <>
            {customDataset && (
              <button
                onClick={() => { setCustomDataset(null); buildForestWithData(heartData, heartMeta.features, heartMeta.targetCol); }}
                style={{ ...inp, fontSize: 10, padding: "3px 9px", cursor: "pointer" }}
              >
                ↩ Heart Disease
              </button>
            )}
            <label style={{
              padding: "4px 10px", borderRadius: 5, border: `1px solid ${C.border}`,
              background: "#1a1f2e", color: C.dim, fontSize: 10, cursor: "pointer",
              fontFamily: "inherit", display: "inline-block",
            }}>
              ↑ Upload CSV
              <input type="file" accept=".csv" style={{ display: "none" }}
                onChange={e => { openFile(e.target.files[0]); e.target.value = ""; }} />
            </label>
          </>
        }
      />

      {/* Controls toolbar */}
      <div style={{
        display: "flex", alignItems: "flex-end", gap: 10,
        padding: "10px 16px",
        background: C.panel,
        borderBottom: `1px solid ${C.border}`,
        flexWrap: "wrap",
      }}>
        {/* Model hyperparameters */}
        <div style={{ display: "flex", alignItems: "flex-end", gap: 10, flex: 1, flexWrap: "wrap" }}>
          <div style={{ display: "flex", flexDirection: "column", gap: 3 }}>
            <label style={{ fontSize: 9, color: C.dim, textTransform: "uppercase", letterSpacing: 1 }}>max_depth</label>
            <input type="number" min={1} max={activeFeatures.length} value={maxDepthStr} disabled={growing}
              onChange={e => setMaxDepthStr(e.target.value)}
              onBlur={() => {
                const v = Math.max(1, Math.min(activeFeatures.length, parseInt(maxDepthStr, 10) || 1));
                setMaxDepth(v); setMaxDepthStr(String(v));
              }}
              style={{ ...inp, width: 54 }} />
          </div>
          {!lockedMaxFeatures && (
            <div style={{ display: "flex", flexDirection: "column", gap: 3 }}>
              <label style={{ fontSize: 9, color: C.dim, textTransform: "uppercase", letterSpacing: 1 }}>max_features</label>
              <select value={featureSubset} disabled={growing} onChange={e => setFeatureSubset(e.target.value)} style={{ ...inp, cursor: "pointer" }}>
                {Object.entries(FEATURE_SUBSET_OPTIONS).map(([k, v]) => (
                  <option key={k} value={k}>{v.label} → {v.fn(activeFeatures.length)}</option>
                ))}
              </select>
            </div>
          )}
          {!lockedNEstimators && (
            <div style={{ display: "flex", flexDirection: "column", gap: 3 }}>
              <label style={{ fontSize: 9, color: C.dim, textTransform: "uppercase", letterSpacing: 1 }}>n_estimators</label>
              <input type="number" min={1} max={100} value={nEstimatorsStr} disabled={growing}
                onChange={e => setNEstimatorsStr(e.target.value)}
                onBlur={() => {
                  const v = Math.max(1, Math.min(100, parseInt(nEstimatorsStr, 10) || 1));
                  setNEstimators(v); setNEstimatorsStr(String(v));
                }}
                style={{ ...inp, width: 54 }} />
            </div>
          )}
          <div style={{ display: "flex", flexDirection: "column", gap: 3 }}>
            <label style={{ fontSize: 9, color: C.dim, textTransform: "uppercase", letterSpacing: 1 }}>speed</label>
            <select value={speed} onChange={e => setSpeed(+e.target.value)} style={{ ...inp, cursor: "pointer", width: 60 }}>
              {[0.5, 1, 2, 4].map(s => <option key={s} value={s}>{s}×</option>)}
            </select>
          </div>
        </div>

        {/* Action buttons */}
        <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
          <button onClick={buildForest} style={{
            padding: "7px 14px", borderRadius: 6, border: `1px solid ${C.border}`,
            background: "#1a1f2e", color: C.text, fontSize: 11,
            fontFamily: "inherit", cursor: "pointer", fontWeight: 600,
          }}>↻ Rebuild</button>
          <button onClick={() => {
            if (growing) { cancelRef.current = true; growRef.current = false; setGrowing(false); }
            else autoGrow(curTree);
          }} style={{
            padding: "7px 20px", borderRadius: 6, border: "none",
            background: growing ? `linear-gradient(135deg,${C.red},#dc2626)` : `linear-gradient(135deg,${C.accent},${C.green})`,
            color: "#000", fontSize: 12, fontFamily: "inherit", cursor: "pointer", fontWeight: 800,
          }}>
            {growing ? "■ Stop" : "▶ Grow"}
          </button>
        </div>
      </div>

      {/* Unified navigation bar */}
      <div style={{
        display: "flex", alignItems: "center",
        padding: "0 16px", borderBottom: `1px solid ${C.border}`,
        background: C.bg, minHeight: 38, overflow: "hidden",
      }}>
        {/* Scrollable tabs area — only this region scrolls */}
        {nEstimators > 1 && (
          <div style={{
            flex: 1, display: "flex", alignItems: "center", gap: 3,
            overflowX: "auto", minWidth: 0, padding: "6px 0",
            /* hide scrollbar visually but keep it functional */
            scrollbarWidth: "none", msOverflowStyle: "none",
          }}>
            {trees.map((_, i) => {
              const s    = treeStates[i];
              const done = s && s.stepIdx >= getSteps(i).length - 1 && getSteps(i).length > 0;
              return (
                <button key={i}
                  onClick={() => { if (!growing) setCurTree(i); }}
                  onDoubleClick={() => { if (!growing) { setCurTree(i); instantComplete(i); } }}
                  title="Click to view · Double-click to instantly complete"
                  style={{
                    padding: "3px 10px", borderRadius: 4, border: "none", flexShrink: 0,
                    background: i === curTree ? C.accent : done ? `${C.green}33` : "#1a1f2e",
                    color: i === curTree ? "#000" : done ? C.green : C.dim,
                    fontSize: 10, fontFamily: "inherit", cursor: "pointer",
                    fontWeight: i === curTree ? 700 : 400,
                  }}>
                  T{i + 1}{done ? " ✓" : ""}
                </button>
              );
            })}
          </div>
        )}
        {nEstimators === 1 && <div style={{ flex: 1 }} />}

        {/* Pinned right side — never scrolls: Complete all | step controls | bootstrap | Reset */}
        <div style={{
          display: "flex", alignItems: "center", gap: 8,
          flexShrink: 0, paddingLeft: 10, paddingTop: 6, paddingBottom: 6,
          borderLeft: nEstimators > 1 ? `1px solid ${C.border}` : "none",
          marginLeft: nEstimators > 1 ? 6 : 0,
        }}>
          {nEstimators > 1 && (
            <button onClick={growAllInstant} disabled={growing} style={{
              padding: "3px 10px", borderRadius: 4, flexShrink: 0,
              border: `1px solid ${C.border}`, background: "none",
              color: C.dim, fontSize: 9, fontFamily: "inherit", cursor: "pointer",
            }}>Complete all</button>
          )}

          {/* Step controls */}
          <button disabled={growing || ts.stepIdx <= -1} onClick={() => goToStep(curTree, ts.stepIdx - 1)}
            style={{ ...inp, cursor: growing || ts.stepIdx <= -1 ? "default" : "pointer", padding: "3px 10px", opacity: growing || ts.stepIdx <= -1 ? 0.3 : 1, fontSize: 11 }}>◀</button>
          <span style={{ fontSize: 10, color: C.dim, minWidth: 72, textAlign: "center", userSelect: "none" }}>
            {ts.stepIdx === -1 ? "ready" : `${ts.stepIdx + 1} / ${totalSteps}`}
          </span>
          <button disabled={growing || ts.stepIdx >= totalSteps - 1} onClick={() => goToStep(curTree, ts.stepIdx + 1)}
            style={{ ...inp, cursor: growing || ts.stepIdx >= totalSteps - 1 ? "default" : "pointer", padding: "3px 10px", opacity: growing || ts.stepIdx >= totalSteps - 1 ? 0.3 : 1, fontSize: 11 }}>▶</button>

          {/* Bootstrap info */}
          {curBootstrap && (
            <span style={{ fontSize: 9, color: C.purple, whiteSpace: "nowrap" }}>
              {curBootstrap.inBag}/{TOTAL_SAMPLES} in-bag · {curBootstrap.oob} OOB
              {curBootstrap.oobAccuracy > 0 ? ` · acc=${curBootstrap.oobAccuracy}` : ""}
            </span>
          )}

          {/* Reset */}
          <button onClick={() => setTS(curTree, EMPTY_TS)} disabled={growing}
            style={{ ...inp, cursor: "pointer", padding: "3px 10px", fontSize: 10, opacity: growing ? 0.3 : 1 }}>Reset</button>
        </div>
      </div>

      {/* SVG canvas */}
      <div ref={canvasRef}
        onMouseDown={onMouseDown} onMouseMove={onMouseMove}
        onMouseUp={onMouseUp} onMouseLeave={onMouseUp}
        style={{
          overflow: "hidden", cursor: cursorGrabbing ? "grabbing" : "grab",
          height: Math.min(500, svgH + 40), background: "#080c14",
          borderBottom: `1px solid ${C.border}`, position: "relative",
        }}>

        {/* Empty state placeholder */}
        {!currentNode && (
          <div style={{
            position: "absolute", inset: 0, display: "flex", flexDirection: "column",
            alignItems: "center", justifyContent: "center", pointerEvents: "none",
          }}>
            <svg width="48" height="44" viewBox="0 0 48 44" fill="none" opacity={0.3}>
              <circle cx="24" cy="10" r="8" stroke={C.dim} strokeWidth="1.5" />
              <circle cx="10" cy="33" r="7" stroke={C.dim} strokeWidth="1.5" />
              <circle cx="38" cy="33" r="7" stroke={C.dim} strokeWidth="1.5" />
              <line x1="24" y1="18" x2="13" y2="26" stroke={C.dim} strokeWidth="1.3" />
              <line x1="24" y1="18" x2="35" y2="26" stroke={C.dim} strokeWidth="1.3" />
            </svg>
            <div style={{ fontSize: 11, color: C.dimmer, marginTop: 14, textAlign: "center", lineHeight: 1.7 }}>
              Press <span style={{ color: C.dim }}>▶ Grow</span> or use <span style={{ color: C.dim }}>→</span> arrow key to start building the tree
            </div>
          </div>
        )}

        <svg width={treeWidth} height={svgH}
          style={{ transform: `translate(${pan.x}px,${pan.y}px) scale(${zoom})`, transformOrigin: "0 0", display: "block" }}>
          <defs>
            <filter id="ng"><feDropShadow dx="0" dy="0" stdDeviation="3" floodColor={C.accent} floodOpacity="0.4" /></filter>
            <filter id="ng-p"><feDropShadow dx="0" dy="0" stdDeviation="4" floodColor={C.orange} floodOpacity="0.55" /></filter>
            <filter id="lg"><feDropShadow dx="0" dy="0" stdDeviation="3" floodColor={C.blue}   floodOpacity="0.3" /></filter>
          </defs>
          {allNodes.map(node => {
            if (node.type !== "split") return null;
            const show = visibleSet.has(node.id);
            return (
              <g key={node.id}>
                <Edge p1={positions[node.id]} p2={positions[node.left?.id]}  visible={show} label="≤" />
                <Edge p1={positions[node.id]} p2={positions[node.right?.id]} visible={show} label=">" />
              </g>
            );
          })}
          {allNodes.map(node => {
            const show  = visibleSet.has(node.id);
            const phase = ts.nodeId === node.id ? ts.phase : show ? 2 : 0;
            return <TreeNode key={node.id} node={node} show={show || ts.nodeId === node.id} phase={phase} pos={positions[node.id]} labelA={classLabels.A} labelB={classLabels.B} />;
          })}
        </svg>

        {/* Floating zoom controls (bottom-right) */}
        <div style={{
          position: "absolute", bottom: 12, right: 12, zIndex: 10,
          display: "flex", flexDirection: "column", alignItems: "center", gap: 1,
          background: "rgba(10,14,23,0.88)", border: `1px solid ${C.border}`,
          borderRadius: 8, padding: "5px 4px",
          backdropFilter: "blur(6px)",
          boxShadow: "0 4px 16px rgba(0,0,0,0.5)",
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

      {/* Feature pool */}
      <div style={{ padding: "10px 16px 4px" }}>
        <div style={{ background: "#0d1117", borderRadius: 10, border: `1px solid ${C.border}`, padding: "10px 14px" }}>
          <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 8, flexWrap: "wrap" }}>
            <span style={{ fontSize: 9, color: C.dim, textTransform: "uppercase", letterSpacing: 1, fontWeight: 600 }}>Feature Pool</span>
            <span style={{ fontSize: 9, color: C.dimmer }}>
              <span style={{ color: C.dimmer }}>● not sampled</span>{"  "}
              <span style={{ color: C.orange }}>● candidate</span>{"  "}
              <span style={{ color: C.green }}>● chosen</span>
            </span>
            {currentNode?.type === "split" && currentNode.globalBestIdx !== currentNode.featureIndex && ts.phase >= 2 && (
              <span style={{ fontSize: 9, color: C.red, marginLeft: "auto" }}>⚠ true best not in subset</span>
            )}
          </div>
          <div style={{ display: "flex", flexWrap: "wrap", gap: 5, justifyContent: "center" }}>
          {activeFeatures.map((f, i) => {
            const cn          = currentNode?.type === "split" ? currentNode : null;
            const isCand      = cn?.candidateIndices?.includes(i);
            const isBest      = cn && i === cn.featureIndex;
            const isGlobalBest= cn && i === cn.globalBestIdx && cn.globalBestIdx !== cn.featureIndex;
            const ev          = cn?.allFeatureEvals?.find(e => e.featureIndex === i);
            const showGini    = ts.phase >= 1 && cn && ev;
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
      </div>

      {/* Calculations panel */}
      <div style={{ padding: "6px 16px 8px" }}>
        <div style={{ fontSize: 9, color: C.dim, marginBottom: 4, textTransform: "uppercase", letterSpacing: 1 }}>Calculations</div>
        <div style={{ padding: 12, background: "#0d1117", borderRadius: 8, border: `1px solid ${C.border}`, fontSize: 10.5, lineHeight: 1.7 }}>
          {!currentNode && <span style={{ color: C.dim }}>Press ▶ Grow or → arrow key to begin…</span>}
          {currentNode?.type === "leaf" && (
            <span>🍃 <strong style={{ color: currentNode.prediction === "A" ? C.blue : C.leafB }}>Leaf</strong> — Predict: <strong>{currentNode.prediction === "A" ? classLabels.A : classLabels.B}</strong> | Gini={currentNode.impurity.toFixed(4)} | n={currentNode.samples} [{currentNode.classA} {classLabels.A}, {currentNode.classB} {classLabels.B}]</span>
          )}
          {currentNode?.type === "split" && (<>
            <div style={{ color: C.accent, fontWeight: 700, marginBottom: 5, fontSize: 11 }}>▸ Node depth={currentNode.depth}, n={currentNode.samples}</div>
            {ts.phase === 0 && (
              <div style={{ color: C.orange, opacity: 0.75 }}>Sampling feature candidates…</div>
            )}
            {ts.phase >= 1 && (
              <div><span style={{ color: C.orange }}>① Random subset ({currentNode.candidateIndices.length}/{activeFeatures.length}):</span> [{currentNode.candidateIndices.map(i => activeFeatures[i]).join(", ")}]</div>
            )}
            {ts.phase >= 1 && (
              <div style={{ marginTop: 3 }}>
                <span style={{ color: C.orange }}>② Gini per candidate:</span>
                <div style={{ marginLeft: 14, marginTop: 2 }}>
                  {currentNode.allFeatureEvals
                    .filter(ev => currentNode.candidateIndices.includes(ev.featureIndex))
                    .map((ev, j) => (
                      <div key={j} style={{ color: ts.phase >= 2 && ev.featureIndex === currentNode.featureIndex ? C.green : C.dim }}>
                        {activeFeatures[ev.featureIndex]}: t={ev.threshold} G={ev.gini.toFixed(4)}
                        {ts.phase >= 2 && ev.featureIndex === currentNode.featureIndex && " ◀ best in subset"}
                      </div>
                    ))}
                </div>
              </div>
            )}
            {ts.phase >= 2 && currentNode.globalBestIdx !== currentNode.featureIndex && (
              <div style={{ marginTop: 3, color: C.red, fontSize: 10 }}>
                ⚠ Global best was <strong>{activeFeatures[currentNode.globalBestIdx]}</strong> (G={currentNode.globalBestGini.toFixed(4)}) but it wasn't in the random subset
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
        <div style={{ padding: "0 16px 16px" }}>
          <div style={{ fontSize: 9, color: C.dim, marginBottom: 4, textTransform: "uppercase", letterSpacing: 1 }}>
            Ensemble — Majority Vote
          </div>
          <div style={{ padding: 14, background: "#0d1117", borderRadius: 8, border: `1px solid ${C.border}` }}>
            {completedTrees.length === 0 ? (
              <span style={{ fontSize: 10.5, color: C.dim }}>Grow trees to see the ensemble prediction. Each tree votes, majority wins.</span>
            ) : (
              <div>
                <div style={{ display: "flex", gap: 6, marginBottom: 10, flexWrap: "wrap", alignItems: "center" }}>
                  {trees.map((_, i) => {
                    const ct = completedTrees.find(t => t.idx === i);
                    // Abbreviate label to fit the compact card (≤6 chars)
                    const abbrev = (label) => label.length <= 10 ? label : label.slice(0, 9) + "…";
                    const cardLabel = ct ? abbrev(ct.prediction === "A" ? classLabels.A : classLabels.B) : "—";
                    const cardColor = ct ? (ct.prediction === "A" ? C.blue : C.leafB) : C.dimmer;
                    return (
                      <div key={i} style={{
                        minWidth: 42, height: 38, borderRadius: 6, display: "flex",
                        flexDirection: "column", alignItems: "center", justifyContent: "center",
                        padding: "0 6px",
                        background: ct ? (ct.prediction === "A" ? C.blue + "22" : C.leafB + "22") : "#151a24",
                        border: `1px solid ${ct ? (ct.prediction === "A" ? C.blue + "66" : C.leafB + "66") : C.border}`,
                        transition: "all .3s",
                      }}>
                        <div style={{ fontSize: 7, color: C.dim }}>T{i + 1}</div>
                        <div style={{ fontSize: 9, fontWeight: 700, color: cardColor, whiteSpace: "nowrap" }}>
                          {cardLabel}
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
                      <div style={{ fontSize: 16, fontWeight: 800, color: ensemblePrediction === "A" ? C.blue : C.leafB }}>
                        {ensemblePrediction === "A" ? classLabels.A : classLabels.B}
                      </div>
                    </div>
                  )}
                </div>
                <div style={{ display: "flex", gap: 16, fontSize: 10, color: C.dim }}>
                  <span>Votes: <strong style={{ color: C.blue }}>{votesA}× {classLabels.A}</strong> vs <strong style={{ color: C.leafB }}>{votesB}× {classLabels.B}</strong></span>
                  <span>Completed: {completedTrees.length}/{nEstimators}</span>
                  {bootstrapInfo.length > 0 && bootstrapInfo.some(b => b.oobAccuracy > 0) && (
                    <span style={{ color: C.purple }}>
                      Avg OOB acc: {(bootstrapInfo.reduce((s, b) => s + b.oobAccuracy, 0) / bootstrapInfo.length).toFixed(3)}
                    </span>
                  )}
                </div>
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
      <div style={{ padding: "8px 16px", borderTop: `1px solid ${C.border}`, fontSize: 8.5, color: C.dim, textAlign: "center" }}>
        max_features={subsetSize}/{activeFeatures.length} · Trees: {nEstimators} · Bootstrap n={TOTAL_SAMPLES}
      </div>
    </div>
  );
}
