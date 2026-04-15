import { useState, useEffect, useRef, useCallback, useMemo } from "react";
import { useLocation } from "react-router-dom";
import Papa from "papaparse";
import { C } from "./theme";
import GlobalHeader from "./GlobalHeader";
import { heartData, heartMeta } from "./data/heartDisease";
import { musicData, musicMeta } from "./data/musicData";
import { salaryData, salaryMeta } from "./data/salary";
import { predictRow } from "./cartAlgorithm";
import TreeWorker from "./treeWorker.js?worker";

// ─── Feature-subset options ────────────────────────────────────────────────────
const FEATURE_SUBSET_OPTIONS = {
  sqrt: { label: "√p (sqrt)", fn: (p) => Math.max(1, Math.round(Math.sqrt(p))) },
  log2: { label: "log₂(p)",  fn: (p) => Math.max(1, Math.round(Math.log2(p))) },
  half: { label: "p/2",       fn: (p) => Math.max(1, Math.round(p / 2)) },
  all:  { label: "p (all)",   fn: (p) => p },
};

// ─── Class color palette ───────────────────────────────────────────────────────
// Index 0 = blue, index 1 = pink — preserves binary heart-disease appearance.
// For 3+ classes the remaining slots provide visually distinct hues.
const CLASS_PALETTE = [
  C.blue, C.leafB, C.green, C.accent, C.purple,
  "#06b6d4", "#f472b6", "#84cc16", "#fb923c", "#a3e635",
];

function classColor(cls, allClasses) {
  const idx = allClasses.indexOf(cls);
  return idx >= 0 ? CLASS_PALETTE[idx % CLASS_PALETTE.length] : C.dim;
}

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

// Aggregate across all leaves: majority class (classification) or weighted mean (regression).
function getTreePrediction(node) {
  const leaves = flattenNodes(node).filter(n => n.type === "leaf");
  if (!leaves.length) return "?";
  // Regression: weighted mean of leaf means
  if (leaves[0].mean !== undefined) {
    const totalN = leaves.reduce((s, l) => s + l.samples, 0);
    return totalN > 0 ? leaves.reduce((s, l) => s + l.mean * l.samples, 0) / totalN : 0;
  }
  // Classification: majority class
  const totals = {};
  leaves.forEach(l => {
    Object.entries(l.classCounts ?? {}).forEach(([cls, cnt]) => {
      totals[cls] = (totals[cls] ?? 0) + cnt;
    });
  });
  return Object.entries(totals).sort((a, b) => b[1] - a[1])[0]?.[0] ?? "?";
}

// Format a split threshold for display on edges: 3 dp for small values, fewer for large.
function fmtThresh(v) {
  if (v == null) return "?";
  const abs = Math.abs(v);
  if (abs >= 1000) return v.toFixed(0);
  if (abs >= 10)   return v.toFixed(1);
  if (abs >= 1)    return v.toFixed(2);
  return v.toFixed(3);
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

// Format a regression prediction value compactly for display.
function formatRegVal(v) {
  if (v === null || v === undefined || (typeof v === "number" && isNaN(v))) return "?";
  const abs = Math.abs(v);
  if (abs >= 1e6)  return v.toLocaleString(undefined, { maximumFractionDigits: 0 });
  if (abs >= 1000) return v.toLocaleString(undefined, { maximumFractionDigits: 1 });
  if (abs >= 10)   return v.toFixed(2);
  return v.toFixed(3);
}

// Auto-detect whether a target column looks like regression (≥20 unique numeric values).
function detectTaskType(rawRows, targetCol) {
  const vals = rawRows.map(r => String(r[targetCol] ?? "").trim());
  const hasNonNumeric = vals.some(v => v === "" || isNaN(parseFloat(v)));
  if (hasNonNumeric) return "classification";
  return new Set(vals).size >= 20 ? "regression" : "classification";
}

// Returns ordered array of node IDs from root to the leaf a sample lands in.
function getSamplePath(node, row, features) {
  const path = [node.id];
  if (node.type === "leaf") return path;
  const fname = features[node.featureIndex];
  const goLeft = row[fname] <= node.threshold;
  return [...path, ...getSamplePath(goLeft ? node.left : node.right, row, features)];
}

// Formats a data row as a compact label for the sample dropdown.
function formatSampleLabel(row, features, idx) {
  const MAX = 5;
  const shown = features.slice(0, MAX);
  const parts = shown.map(f => {
    const short = f.length > 7 ? f.slice(0, 6) + "…" : f;
    const v = row[f];
    return `${short}=${v}`;
  });
  const extra = features.length > MAX ? ` +${features.length - MAX} more` : "";
  return `#${idx + 1}: ${parts.join(" · ")}${extra}`;
}

// ─── SVG components ────────────────────────────────────────────────────────────
const PATH_COLOR = "#22d3ee";

function Edge({ p1, p2, visible, label, onPath, sampleActive }) {
  if (!visible || !p1 || !p2) return null;
  const highlighted = sampleActive && onPath;
  const dimmed      = sampleActive && !onPath;
  // Line endpoints (clear of node circles/boxes)
  const x1 = p1.x, y1 = p1.y + 27;
  const x2 = p2.x, y2 = p2.y - 24;
  // Place label 38% from the parent end — upper portion, clear of child node
  const lx = x1 + (x2 - x1) * 0.38;
  const ly = y1 + (y2 - y1) * 0.38;
  const labelW = label.length * 5.6 + 10;
  const textCol = highlighted ? PATH_COLOR : C.muted;
  return (
    <g style={{ opacity: dimmed ? 0.15 : 1, transition: "opacity .3s" }}>
      <line x1={x1} y1={y1} x2={x2} y2={y2}
        stroke={highlighted ? PATH_COLOR : C.edge}
        strokeWidth={highlighted ? 2.4 : 1.4}
        strokeDasharray={highlighted ? undefined : "4 2"}
        style={{ transition: "stroke .25s, stroke-width .25s" }}
      />
      {/* Background pill so label is legible over the dashed line */}
      <rect x={lx - labelW / 2} y={ly - 7} width={labelW} height={13}
        rx={3} fill={C.bg} opacity={0.88} />
      <text x={lx} y={ly + 2}
        textAnchor="middle" dominantBaseline="middle"
        fill={textCol} fontSize={9} fontFamily="'JetBrains Mono',monospace"
        fontWeight={highlighted ? 700 : 400}>
        {label}
      </text>
    </g>
  );
}

function TreeNode({ node, show, phase, pos, allClasses, onPath, sampleActive, isRegression }) {
  if (!show || !pos) return null;
  const { x, y } = pos;
  const highlighted = sampleActive && onPath;
  const dimmed      = sampleActive && !onPath;

  // ── Leaf node ─────────────────────────────────────────────────────────────
  if (node.type === "leaf") {
    const vis = phase >= 1 ? 1 : 0;
    const gStyle = { opacity: dimmed ? 0.2 : vis, transition: "opacity .45s ease-out", transformOrigin: `${x}px ${y}px`, animation: vis ? "nodeIn 0.35s ease-out" : "none" };

    // ── Regression leaf ──────────────────────────────────────────────────────
    if (node.mean !== undefined) {
      const leafCol = highlighted ? PATH_COLOR : C.blue;
      const meanStr = formatRegVal(node.mean);
      const minStr  = node.min != null ? formatRegVal(node.min) : "?";
      const maxStr  = node.max != null ? formatRegVal(node.max) : "?";
      return (
        <g style={gStyle}>
          <rect x={x - 40} y={y - 26} width={80} height={54} rx={11}
            fill={C.panel} stroke={highlighted ? PATH_COLOR : `${C.blue}88`}
            strokeWidth={highlighted ? 2 : 1.4} filter={highlighted ? "url(#path-glow)" : "url(#lg)"} />
          <text x={x} y={y - 11} textAnchor="middle" fill={leafCol} fontSize={9.5}
            fontFamily="'JetBrains Mono',monospace" fontWeight={700}>{meanStr}</text>
          <text x={x} y={y + 3} textAnchor="middle" fill={C.dim} fontSize={7.5}
            fontFamily="'JetBrains Mono',monospace">n={node.samples}</text>
          <text x={x} y={y + 16} textAnchor="middle" fill={C.dim} fontSize={7}
            fontFamily="'JetBrains Mono',monospace">{minStr}–{maxStr}</text>
        </g>
      );
    }

    // ── Classification leaf ──────────────────────────────────────────────────
    const predColor  = classColor(node.prediction, allClasses);
    const displayLabel = typeof node.prediction === "string" && node.prediction.length > 11
      ? node.prediction.slice(0, 10) + "…" : String(node.prediction);
    const leafStroke = highlighted ? PATH_COLOR : `${predColor}88`;
    const leafFilter = highlighted ? "url(#path-glow)" : "url(#lg)";

    const BAR_W = 56, barX0 = x - 28;
    let xCursor = barX0;
    const barSegs = allClasses.map(cls => {
      const count = node.classCounts?.[cls] ?? 0;
      const w = node.samples > 0 ? Math.max(0, (count / node.samples) * BAR_W) : 0;
      const rx = xCursor; xCursor += w;
      return w > 0 ? <rect key={cls} x={rx} y={y + 2} width={w} height={5} fill={classColor(cls, allClasses)} /> : null;
    });

    return (
      <g style={gStyle}>
        <rect x={x - 38} y={y - 24} width={76} height={50} rx={11}
          fill={C.panel} stroke={leafStroke} strokeWidth={highlighted ? 2 : 1.4} filter={leafFilter} />
        <text x={x} y={y - 9} textAnchor="middle" fill={highlighted ? PATH_COLOR : C.text} fontSize={9}
          fontFamily="'JetBrains Mono',monospace" fontWeight={600}>{displayLabel}</text>
        <rect x={barX0} y={y + 2} width={BAR_W} height={5} rx={2.5} fill="rgba(255,255,255,0.06)" />
        {barSegs}
        <text x={x} y={y + 17} textAnchor="middle" fill={C.dim} fontSize={7.5}
          fontFamily="'JetBrains Mono',monospace">n={node.samples} G={node.impurity.toFixed(3)}</text>
      </g>
    );
  }

  // ── Split node ─────────────────────────────────────────────────────────────
  const revealed    = phase >= 2;
  const borderColor = highlighted ? PATH_COLOR : (revealed ? C.accent : C.orange);
  const filterId    = highlighted ? "url(#path-glow)" : (revealed ? "url(#ng)" : "url(#ng-p)");
  const { line1, line2 } = splitFeatureName(node.featureName ?? "");
  const crit = isRegression ? "MSE" : "G";

  return (
    <g style={{ opacity: dimmed ? 0.2 : 1, transition: "opacity .3s" }}>
      <rect x={x - 90} y={y - 27} width={180} height={54} rx={11}
        fill={C.panel} stroke={borderColor} strokeWidth={revealed ? 1.4 : 1.8}
        filter={filterId} style={{ transition: "stroke 0.25s, stroke-width 0.25s" }} />
      <g style={{ opacity: revealed ? 0 : 1, transition: "opacity 0.2s" }}>
        <text x={x} y={y - 5} textAnchor="middle" fill={C.orange} fontSize={11}
          fontFamily="'JetBrains Mono',monospace" fontWeight={700}>?</text>
        <text x={x} y={y + 11} textAnchor="middle" fill={C.dim} fontSize={7.5}
          fontFamily="'JetBrains Mono',monospace">n={node.samples}</text>
      </g>
      <g style={{ opacity: revealed ? 1 : 0, transition: "opacity 0.3s" }}>
        {line2 ? (
          <>
            <text x={x} y={y - 13} textAnchor="middle" fill={C.accent} fontSize={8.5}
              fontFamily="'JetBrains Mono',monospace" fontWeight={700}>{line1}</text>
            <text x={x} y={y - 2} textAnchor="middle" fill={C.accent} fontSize={8.5}
              fontFamily="'JetBrains Mono',monospace" fontWeight={700}>{line2}</text>
            <text x={x} y={y + 14} textAnchor="middle" fill={C.dim} fontSize={7.5}
              fontFamily="'JetBrains Mono',monospace">≤{node.threshold} {crit}={node.gini.toFixed(3)} n={node.samples}</text>
          </>
        ) : (
          <>
            <text x={x} y={y - 6} textAnchor="middle" fill={C.accent} fontSize={9}
              fontFamily="'JetBrains Mono',monospace" fontWeight={700}>{line1}</text>
            <text x={x} y={y + 10} textAnchor="middle" fill={C.dim} fontSize={7.5}
              fontFamily="'JetBrains Mono',monospace">≤{node.threshold} {crit}={node.gini.toFixed(3)} n={node.samples}</text>
          </>
        )}
      </g>
    </g>
  );
}

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
    if (r) setPos({ left: r.left + r.width / 2, top: r.bottom + 8 });
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
      {pos && (
        <div style={{
          position: "fixed", left: pos.left, top: pos.top,
          transform: "translateX(-50%)", zIndex: 1000,
          background: "#141c2e", border: `1px solid rgba(255,255,255,0.08)`,
          borderRadius: 10, padding: "12px 16px",
          fontSize: 10.5, color: C.dim, lineHeight: 1.75,
          maxWidth: 280, pointerEvents: "none",
          boxShadow: "0 8px 32px rgba(0,0,0,0.6)",
        }}>
          {lines.map((l, i) => <div key={i}>{l}</div>)}
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

// selectedCols: array of column names the user kept (includes target). null = use all headers.
function processCSVData(rawRows, headers, targetCol, naStrategy, sampleMode, selectedCols, taskType = "classification") {
  // Resolve active column set — always include target, apply user selection for features
  const activeCols = selectedCols
    ? [targetCol, ...selectedCols.filter(h => h !== targetCol)]
    : headers;

  // Apply NA strategy over the active columns only
  let rows;
  if (naStrategy === "drop") {
    rows = rawRows.filter(r => activeCols.every(h => !NA_VALS.has(String(r[h] ?? "").trim())));
  } else {
    // Fill non-target numeric active columns with column median
    const medians = {};
    activeCols.filter(h => h !== targetCol).forEach(h => {
      const nums = rawRows.map(r => parseFloat(r[h])).filter(v => !isNaN(v)).sort((a, b) => a - b);
      if (nums.length) medians[h] = nums[Math.floor(nums.length / 2)];
    });
    rows = rawRows.map(r => {
      const copy = { ...r };
      activeCols.forEach(h => {
        if (h !== targetCol && NA_VALS.has(String(r[h] ?? "").trim())) {
          copy[h] = String(medians[h] ?? 0);
        }
      });
      return copy;
    });
  }

  if (rows.length === 0) return null;

  const totalRows = rows.length;
  const sampleN = (typeof sampleMode === "number") ? Math.min(sampleMode, rows.length) : rows.length;
  if (sampleN < rows.length) {
    if (taskType === "classification") {
      rows = stratifiedSample(rows, targetCol, sampleN);
    } else {
      for (let i = rows.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [rows[i], rows[j]] = [rows[j], rows[i]];
      }
      rows = rows.slice(0, sampleN);
    }
  }
  const sampledRows = rows.length;

  const featCols = activeCols.filter(h => h !== targetCol);

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
      catVals[h].slice(1).forEach(v => features.push(`${h}_${v}`));
    }
  });

  // Classification: map target to stable class label strings.
  // Regression: keep target as raw float.
  let classLabels = [];
  let targetMap   = {};
  if (taskType === "classification") {
    const targetUniq = [...new Set(rows.map(r => String(r[targetCol] ?? "")))].sort();
    classLabels = targetUniq.map(v => formatClassLabel(v));
    targetMap   = Object.fromEntries(targetUniq.map((v, i) => [v, classLabels[i]]));
  }

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
    obj["target"] = taskType === "regression"
      ? parseFloat(r[targetCol]) || 0
      : (targetMap[String(r[targetCol] ?? "")] ?? String(r[targetCol] ?? ""));
    return obj;
  });

  return { data, features, targetCol: "target", totalRows, sampledRows, classLabels };
}

// ─── CSV Modal ─────────────────────────────────────────────────────────────────
function DataModal({ modal, onUpdate, onConfirm, onCancel }) {
  const { fileName, rawRows, headers, naStats, selectedTarget, naStrategy, sampleMode, selectedColumns, warning, taskType = "classification" } = modal;
  const includedCols = selectedColumns ?? headers;
  const [previewOpen, setPreviewOpen] = useState(true);

  const toggleCol = (h) => {
    if (h === selectedTarget) return;
    const next = includedCols.includes(h)
      ? includedCols.filter(c => c !== h)
      : [...includedCols, h];
    if (!next.includes(selectedTarget)) next.push(selectedTarget);
    onUpdate({ selectedColumns: next });
  };

  const handleConfirm = () => {
    const result = processCSVData(rawRows, headers, selectedTarget, naStrategy, sampleMode ?? rawRows.length, includedCols, taskType);
    if (!result) return;
    onConfirm(result.data, result.features, result.targetCol,
              fileName.replace(".csv", ""), result.totalRows, result.sampledRows, result.classLabels, taskType, selectedTarget);
  };

  const featureCols  = headers.filter(h => h !== selectedTarget);
  const checkedCount = featureCols.filter(h => includedCols.includes(h)).length;

  // Inject slider CSS once
  useEffect(() => {
    const id = "sampling-slider-style";
    if (document.getElementById(id)) return;
    const s = document.createElement("style");
    s.id = id;
    s.textContent = `
      input.sampling-slider {
        -webkit-appearance: none; appearance: none;
        outline: none; cursor: pointer;
        background: transparent; height: 24px; width: 100%;
      }
      input.sampling-slider::-webkit-slider-runnable-track {
        height: 4px; background: transparent; border-radius: 2px;
      }
      input.sampling-slider::-moz-range-track {
        height: 4px; background: transparent; border-radius: 2px;
      }
      input.sampling-slider::-webkit-slider-thumb {
        -webkit-appearance: none;
        width: 18px; height: 18px; border-radius: 50%;
        background: #f59e0b; cursor: pointer;
        margin-top: -7px; border: none;
        box-shadow: 0 1px 4px rgba(0,0,0,0.5);
        transition: box-shadow 0.15s ease, width 0.15s ease, height 0.15s ease, margin-top 0.15s ease;
      }
      input.sampling-slider:hover::-webkit-slider-thumb {
        width: 20px; height: 20px; margin-top: -8px;
        box-shadow: 0 1px 8px rgba(0,0,0,0.5), 0 0 0 5px rgba(245,158,11,0.18);
      }
      input.sampling-slider::-moz-range-thumb {
        width: 18px; height: 18px; border-radius: 50%;
        background: #f59e0b; cursor: pointer; border: none;
        box-shadow: 0 1px 4px rgba(0,0,0,0.5);
      }
    `;
    document.head.appendChild(s);
    return () => document.getElementById(id)?.remove();
  }, []);

  // ── Per-column stats for header tooltips ────────────────────────────────────
  const colStats = useMemo(() => {
    const stats = {};
    for (const h of headers) {
      const nonNA  = rawRows
        .map(r => String(r[h] ?? "").trim())
        .filter(v => !NA_VALS.has(v));
      const unique = new Set(nonNA).size;
      stats[h] = { unique, missing: naStats.byCol[h] ?? 0 };
    }
    return stats;
  }, [rawRows, headers, naStats]);

  const [colTooltip, setColTooltip] = useState(null); // { col, x, y }

  // ── Shared style tokens ──────────────────────────────────────────────────────
  const sectionLabel = {
    fontSize: 10, color: C.text, fontWeight: 600,
    display: "block", marginBottom: 8,
  };
  // iOS-style segmented pill track
  const segTrack = {
    display: "flex", background: "rgba(255,255,255,0.05)",
    borderRadius: 100, padding: 3,
  };
  const segBtn = (active) => ({
    flex: 1, padding: "7px 12px", borderRadius: 100, border: "none",
    cursor: "pointer", fontSize: 10,
    fontWeight: active ? 700 : 400,
    background: active ? C.accent : "transparent",
    color: active ? "#000" : C.dim,
    transition: "background 0.2s ease-out, color 0.2s ease-out",
  });

  return (
    <div style={{
      position: "fixed", inset: 0, background: "rgba(4,8,18,0.82)", zIndex: 100,
      display: "flex", alignItems: "center", justifyContent: "center",
      backdropFilter: "blur(8px)",
    }}>
      <div style={{
        background: "#12192b",
        borderRadius: 20,
        boxShadow: "0 8px 32px rgba(0,0,0,0.55), 0 0 0 1px rgba(255,255,255,0.06)",
        padding: "28px 28px 24px",
        maxWidth: 500, width: "90vw",
        color: C.text,
        maxHeight: "88vh", overflowY: "auto",
        scrollbarWidth: "thin", scrollbarColor: `${C.border} transparent`,
      }}>

        {/* Title */}
        <div style={{ fontSize: 14, fontWeight: 700, color: C.text, marginBottom: 4 }}>
          Configure dataset
        </div>
        <div style={{ fontSize: 10, color: C.dim, marginBottom: 22, lineHeight: 1.7 }}>
          <span style={{ color: C.accent, fontWeight: 600 }}>{fileName}</span>
          {"  ·  "}{rawRows.length.toLocaleString()} rows · {headers.length} columns
          {naStats.total > 0 && <span style={{ color: C.orange, marginLeft: 8 }}>⚠ {naStats.total} missing</span>}
          {warning && <div style={{ color: C.red, marginTop: 3, fontSize: 9 }}>⚠ {warning}</div>}
        </div>

        {/* ── Preview ────────────────────────────────────────────────────────── */}
        <div style={{ marginBottom: 22 }}>
          <button
            onClick={() => setPreviewOpen(o => !o)}
            style={{
              display: "flex", alignItems: "center", gap: 7,
              width: "100%", cursor: "pointer",
              padding: "8px 12px", borderRadius: 9,
              background: previewOpen ? "rgba(245,158,11,0.08)" : "rgba(255,255,255,0.04)",
              border: "none",
              boxShadow: previewOpen
                ? `inset 0 0 0 1px ${C.accent}55`
                : "inset 0 0 0 1px rgba(255,255,255,0.09)",
              fontSize: 10, fontWeight: 500,
              color: previewOpen ? C.accent : C.dim,
              transition: "background 0.15s, box-shadow 0.15s, color 0.15s",
            }}
            onMouseEnter={e => {
              if (!previewOpen) {
                e.currentTarget.style.background = "rgba(255,255,255,0.07)";
                e.currentTarget.style.color = C.text;
              }
            }}
            onMouseLeave={e => {
              e.currentTarget.style.background = previewOpen ? "rgba(245,158,11,0.08)" : "rgba(255,255,255,0.04)";
              e.currentTarget.style.color = previewOpen ? C.accent : C.dim;
            }}
          >
            <span style={{
              fontSize: 10, lineHeight: 1,
              display: "inline-block",
              transition: "transform 0.2s ease-out",
              transform: previewOpen ? "rotate(90deg)" : "rotate(0deg)",
            }}>▶</span>
            Preview &amp; select columns (first 100 rows)
          </button>
          {previewOpen && (
            <>
              <div style={{
                marginTop: 10, overflowX: "auto", overflowY: "auto", height: 290,
                borderRadius: 12,
                background: "#0a0f1a",
                boxShadow: "0 0 0 1px rgba(255,255,255,0.06)",
                scrollbarWidth: "thin", scrollbarColor: `${C.border} transparent`,
              }}>
                <table style={{
                  borderCollapse: "collapse", fontSize: 8.5,
                  fontFamily: "'JetBrains Mono',monospace",
                  whiteSpace: "nowrap", width: "100%",
                  fontVariantNumeric: "tabular-nums",
                }}>
                  <thead>
                    <tr>
                      {headers.map(h => {
                        const isTarget  = h === selectedTarget;
                        const isChecked = isTarget || includedCols.includes(h);
                        return (
                          <th
                            key={h}
                            onClick={() => !isTarget && toggleCol(h)}
                            onMouseEnter={e => {
                              const r = e.currentTarget.getBoundingClientRect();
                              setColTooltip({ col: h, x: r.left, y: r.bottom + 4 });
                            }}
                            onMouseLeave={() => setColTooltip(null)}
                            style={{
                              padding: "7px 10px", textAlign: "left",
                              background: "#0a0f1a",
                              position: "sticky", top: 0,
                              boxShadow: "0 1px 0 rgba(255,255,255,0.08)",
                              whiteSpace: "nowrap",
                              cursor: isTarget ? "default" : "pointer",
                              opacity: isChecked ? 1 : 0.35,
                              transition: "opacity 0.15s",
                              userSelect: "none",
                            }}
                          >
                            <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
                              {/* Checkbox */}
                              <div style={{
                                width: 13, height: 13, borderRadius: 3, flexShrink: 0,
                                background: isChecked ? (isTarget ? C.accent : C.accent) : "transparent",
                                boxShadow: isChecked ? "none" : "0 0 0 1.5px rgba(255,255,255,0.25)",
                                display: "flex", alignItems: "center", justifyContent: "center",
                                transition: "background 0.15s, box-shadow 0.15s",
                                opacity: isTarget ? 0.5 : 1,
                              }}>
                                {isChecked && (
                                  <svg width="8" height="6" viewBox="0 0 8 6" fill="none">
                                    <path d="M1 3L3 5L7 1" stroke="#000" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/>
                                  </svg>
                                )}
                              </div>
                              {/* Label + NA count */}
                              <div style={{ display: "flex", flexDirection: "column", gap: 1 }}>
                                <div style={{ display: "flex", alignItems: "center", gap: 4 }}>
                                  <span style={{
                                    fontWeight: 600, fontSize: 8.5,
                                    color: isTarget ? C.accent : isChecked ? C.dim : C.dimmer,
                                  }}>
                                    {h}
                                  </span>
                                  {isTarget && (
                                    <span style={{ fontSize: 7, color: C.accent, opacity: 0.65, fontWeight: 400 }}>target</span>
                                  )}
                                </div>
                                {(naStats.byCol[h] ?? 0) > 0 && (
                                  <span style={{ fontSize: 7, color: C.orange, opacity: 0.8, fontWeight: 400 }}>
                                    {naStats.byCol[h]} missing
                                  </span>
                                )}
                              </div>
                            </div>
                          </th>
                        );
                      })}
                    </tr>
                  </thead>
                  <tbody>
                    {rawRows.slice(0, 100).map((row, ri) => (
                      <tr key={ri} style={{ background: ri % 2 === 1 ? "rgba(255,255,255,0.025)" : "transparent" }}>
                        {headers.map(h => {
                          const isTarget  = h === selectedTarget;
                          const isChecked = isTarget || includedCols.includes(h);
                          return (
                            <td key={h} style={{
                              padding: "3.5px 10px",
                              color: isTarget ? "#94a3b8" : C.dimmer,
                              fontWeight: isTarget ? 500 : 400,
                              borderBottom: "1px solid rgba(255,255,255,0.025)",
                              opacity: isChecked ? 1 : 0.25,
                              transition: "opacity 0.15s",
                            }}>
                              {String(row[h] ?? "—")}
                            </td>
                          );
                        })}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              {/* Feature selection summary */}
              <div style={{
                marginTop: 7, fontSize: 9, color: C.dimmer,
                display: "flex", alignItems: "center", gap: 6,
              }}>
                <span style={{ color: checkedCount < featureCols.length ? C.accent : C.dimmer, fontWeight: checkedCount < featureCols.length ? 600 : 400 }}>
                  {checkedCount}/{featureCols.length} features selected
                </span>
                {checkedCount < featureCols.length && (
                  <span>· {featureCols.length - checkedCount} excluded</span>
                )}
                <button
                  onClick={() => onUpdate({ selectedColumns: headers })}
                  style={{
                    marginLeft: "auto", background: "none", border: "none",
                    cursor: "pointer", fontSize: 8.5, color: C.dimmer,
                    padding: 0,
                    transition: "color 0.12s",
                  }}
                  onMouseEnter={e => { e.currentTarget.style.color = C.text; }}
                  onMouseLeave={e => { e.currentTarget.style.color = C.dimmer; }}
                >
                  select all
                </button>
              </div>
            </>
          )}
        </div>

        {/* ── Target column ──────────────────────────────────────────────────── */}
        <div style={{ marginBottom: 22 }}>
          <span style={sectionLabel}>Target column</span>
          <div style={{ position: "relative" }}>
            <select
              value={selectedTarget}
              onChange={e => {
                const newTarget = e.target.value;
                const detected  = detectTaskType(rawRows, newTarget);
                onUpdate({ selectedTarget: newTarget, selectedColumns: headers, taskType: detected });
              }}
              onFocus={e => { e.target.style.boxShadow = `0 0 0 2px ${C.accent}44`; }}
              onBlur={e => { e.target.style.boxShadow = "none"; }}
              style={{
                width: "100%", padding: "9px 32px 9px 12px",
                background: "rgba(255,255,255,0.05)", borderRadius: 10,
                border: "none", color: C.text, fontSize: 11,
                fontFamily: "'JetBrains Mono',monospace",
                cursor: "pointer", outline: "none",
                appearance: "none", WebkitAppearance: "none",
                transition: "box-shadow 0.15s",
              }}>
              {headers.map(h => <option key={h} value={h}>{h}</option>)}
            </select>
            <span style={{
              position: "absolute", right: 12, top: "50%", transform: "translateY(-50%)",
              pointerEvents: "none", color: C.dimmer, fontSize: 10, lineHeight: 1,
            }}>▾</span>
          </div>
        </div>

        {/* ── Task type ──────────────────────────────────────────────────────── */}
        <div style={{ marginBottom: 22 }}>
          <span style={sectionLabel}>Task type</span>
          <div style={segTrack}>
            {["classification", "regression"].map(t => (
              <button key={t} onClick={() => onUpdate({ taskType: t })} style={segBtn(taskType === t)}>
                {t === "classification" ? "Classification" : "Regression"}
              </button>
            ))}
          </div>
          <div style={{ fontSize: 9, color: C.dimmer, marginTop: 7 }}>
            {taskType === "regression"
              ? "Predicts a continuous numeric value · MSE split criterion"
              : "Predicts a class label · Gini impurity split criterion"}
          </div>
        </div>



        {/* ── Missing values ─────────────────────────────────────────────────── */}
        {naStats.total > 0 && (
          <div style={{ marginBottom: 22 }}>
            <span style={sectionLabel}>Handle missing values</span>
            <div style={segTrack}>
              {[
                { key: "drop", label: "Drop rows" },
                { key: "median", label: "Fill with median" },
              ].map(({ key, label }) => (
                <button key={key} onClick={() => onUpdate({ naStrategy: key })} style={segBtn(naStrategy === key)}>
                  {label}
                </button>
              ))}
            </div>
          </div>
        )}

        {/* ── Sampling ───────────────────────────────────────────────────────── */}
        {rawRows.length > 100 && (() => {
          const sv        = typeof sampleMode === "number" ? sampleMode : Math.min(1000, rawRows.length);
          const capVal    = Math.min(rawRows.length, 2000);
          const atCap     = sv >= rawRows.length && rawRows.length < 2000;
          const valuePct  = Math.min((sv - 100) / 1900 * 100, 100);
          const capPct    = Math.min((capVal - 100) / 1900 * 100, 100);
          const recommPct = (1000 - 100) / 1900 * 100; // ≈47.4%
          const showRecomm = rawRows.length >= 1000;

          const trackGrad = rawRows.length < 2000
            ? `linear-gradient(to right, ${C.accent} 0% ${valuePct}%, rgba(255,255,255,0.09) ${valuePct}% ${capPct}%, rgba(255,255,255,0.03) ${capPct}% 100%)`
            : `linear-gradient(to right, ${C.accent} 0% ${valuePct}%, rgba(255,255,255,0.09) ${valuePct}% 100%)`;

          return (
            <div style={{ marginBottom: 22 }}>
              {/* Header row */}
              <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 12 }}>
                <span style={sectionLabel}>Sampling</span>
                <span style={{ fontSize: 13, lineHeight: 1 }}>
                  <span style={{ fontWeight: 600, color: C.accent }}>{sv.toLocaleString()}</span>
                  <span style={{ fontSize: 10, color: C.dim, marginLeft: 4 }}>rows</span>
                </span>
              </div>

              {/* Slider */}
              <div style={{ position: "relative", height: 24, marginBottom: 0 }}>
                {/* Track */}
                <div style={{
                  position: "absolute", left: 0, right: 0, top: "50%",
                  transform: "translateY(-50%)", height: 4, borderRadius: 2,
                  background: trackGrad, pointerEvents: "none",
                }} />
                {/* Dataset limit tick */}
                {rawRows.length < 2000 && (
                  <div style={{
                    position: "absolute", top: "50%", left: `${capPct}%`,
                    transform: "translate(-50%, -50%)",
                    width: 1.5, height: 8, borderRadius: 1,
                    background: C.dimmer, pointerEvents: "none", opacity: 0.6,
                  }} />
                )}
                <input
                  type="range"
                  className="sampling-slider"
                  min={100} max={2000} step={100}
                  value={sv}
                  onChange={e => {
                    const v = Math.min(+e.target.value, rawRows.length);
                    onUpdate({ sampleMode: v });
                  }}
                />
              </div>

              {/* Scale labels row with recommended marker */}
              <div style={{ position: "relative", height: 20, marginBottom: 6 }}>
                <span style={{ position: "absolute", left: 0, fontSize: 7.5, color: C.text, top: 4 }}>100</span>
                <span style={{ position: "absolute", right: 0, fontSize: 7.5, color: C.text, top: 4 }}>2,000</span>
                {showRecomm && (
                  <div style={{
                    position: "absolute", top: 0, left: `${recommPct}%`,
                    transform: "translateX(-50%)",
                    display: "flex", flexDirection: "column", alignItems: "center",
                    pointerEvents: "none",
                  }}>
                    <div style={{ width: 1, height: 5, background: C.dim }} />
                    <span style={{ fontSize: 7, color: C.dim, whiteSpace: "nowrap", marginTop: 1 }}>
                      recommended
                    </span>
                  </div>
                )}
              </div>

              {/* Notes */}
              {atCap ? (
                <div style={{ fontSize: 9, color: C.orange }}>
                  Dataset only has {rawRows.length.toLocaleString()} rows — using all data
                </div>
              ) : (
                <div style={{ fontSize: 9, color: C.dimmer }}>
                  Stratified sampling preserves class proportions
                </div>
              )}
            </div>
          );
        })()}

        {/* Privacy note */}
        <div style={{ fontSize: 8.5, color: C.dimmer, marginBottom: 20, textAlign: "center", letterSpacing: "0.02em" }}>
          🔒 Your data never leaves your browser
        </div>

        {/* Actions */}
        <div style={{ display: "flex", gap: 8 }}>
          <button onClick={onCancel}
            onMouseEnter={e => { e.currentTarget.style.color = C.text; e.currentTarget.style.background = "rgba(255,255,255,0.06)"; }}
            onMouseLeave={e => { e.currentTarget.style.color = C.dim; e.currentTarget.style.background = "transparent"; }}
            style={{
              flex: 1, padding: "10px", borderRadius: 12, border: "none",
              background: "transparent", color: C.dim, fontSize: 11,
              fontFamily: "inherit", cursor: "pointer",
              transition: "color 0.15s, background 0.15s",
            }}>Cancel</button>
          <button onClick={handleConfirm}
            onMouseEnter={e => { e.currentTarget.style.transform = "scale(1.02)"; }}
            onMouseLeave={e => { e.currentTarget.style.transform = "scale(1)"; }}
            style={{
              flex: 2, padding: "10px", borderRadius: 12, border: "none",
              background: `linear-gradient(135deg,${C.accent},#d97706)`,
              color: "#000", fontSize: 11, fontFamily: "inherit",
              cursor: "pointer", fontWeight: 700,
              transition: "transform 0.15s ease-out",
            }}>Confirm &amp; Build</button>
        </div>
      </div>

      {/* ── Column header tooltip ──────────────────────────────────────────── */}
      {colTooltip && colStats[colTooltip.col] && (() => {
        const s = colStats[colTooltip.col];
        return (
          <div style={{
            position: "fixed",
            left: Math.min(colTooltip.x, window.innerWidth - 160),
            top: colTooltip.y,
            zIndex: 9999,
            background: "#1a2235",
            borderRadius: 8,
            padding: "8px 11px",
            boxShadow: "0 8px 24px rgba(0,0,0,0.6), inset 0 0 0 1px rgba(255,255,255,0.08)",
            pointerEvents: "none",
          }}>
            <div style={{ display: "flex", flexDirection: "column", gap: 3, fontSize: 9, color: C.dim }}>
              <div style={{ display: "flex", justifyContent: "space-between", gap: 16 }}>
                <span>Unique values</span>
                <span style={{ color: C.text, fontFamily: "'JetBrains Mono',monospace" }}>{s.unique}</span>
              </div>
              <div style={{ display: "flex", justifyContent: "space-between", gap: 16 }}>
                <span>Missing</span>
                <span style={{ color: s.missing > 0 ? C.orange : C.dim, fontFamily: "'JetBrains Mono',monospace" }}>
                  {s.missing}
                </span>
              </div>
            </div>
          </div>
        );
      })()}
    </div>
  );
}

// ─── Main component ────────────────────────────────────────────────────────────
export default function RandomForestViz({ mode = "random-forest" }) {
  const lockedNEstimators = mode === "decision-tree" ? 1 : null;
  const lockedMaxFeatures = (mode === "decision-tree" || mode === "bagging") ? "all" : null;

  // ── Dataset state ──────────────────────────────────────────────────────────
  const [builtinDataset, setBuiltinDataset] = useState("heart"); // "heart" | "music" | "salary"
  const [customDataset, setCustomDataset] = useState(null);
  const fileInputRef = useRef(null);
  const [csvModal, setCsvModal]           = useState(null);
  const [dragOver, setDragOver]           = useState(false);
  const [selectedSampleIdx, setSelectedSampleIdx] = useState(null);
  const [oobTooltipVisible, setOobTooltipVisible] = useState(false);

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
  const [nEstimators, setNEstimators]   = useState(lockedNEstimators ?? 3);
  const [nEstimatorsStr, setNEstimatorsStr] = useState(String(lockedNEstimators ?? 3));
  const [trees, setTrees]               = useState([]);
  const [bootstrapInfo, setBootstrapInfo] = useState([]);
  const [curTree, setCurTree]           = useState(0);
  const [treeStates, setTreeStates]     = useState({});
  const [growing, setGrowing]           = useState(false);
  const [speed, setSpeed]               = useState(1);
  const [buildProgress, setBuildProgress] = useState(null); // null | { done, total }
  const location = useLocation();
  const growRef   = useRef(false);
  const cancelRef = useRef(false);
  const workerRef = useRef(null);

  const [tabTooltip, setTabTooltip]           = useState(null); // { x, y }
  const [dragRange, setDragRange]             = useState(null); // { start, end } | null
  const [hintDismissed, setHintDismissed]     = useState(false);
  const [scrollHint, setScrollHint]           = useState(false); // "Scroll down to predict ↓" hint
  const [lockedParamTooltip, setLockedParamTooltip] = useState(null); // { key, x, y }
  const dragRef    = useRef({ active: false, startIdx: null, endIdx: null, moved: false });
  const tabRefs    = useRef([]);
  const tabScrollRef = useRef(null);
  const predSectionRef = useRef(null);

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
      select.ds-pill {
        -webkit-appearance: none; appearance: none;
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 20px;
        color: #e2e8f0;
        font-family: inherit;
        font-size: 12px;
        font-weight: 500;
        padding: 5px 28px 5px 12px;
        cursor: pointer;
        outline: none;
        transition: box-shadow 0.2s ease, border-color 0.2s ease;
        min-width: 0;
      }
      select.ds-pill:hover {
        border-color: rgba(255,255,255,0.2);
        box-shadow: 0 0 0 3px rgba(245,158,11,0.12);
      }
      select.ds-pill:focus {
        border-color: rgba(245,158,11,0.4);
        box-shadow: 0 0 0 3px rgba(245,158,11,0.12);
      }
      select.ds-pill option {
        background: #111827;
        color: #e2e8f0;
      }
    `;
    document.head.appendChild(s);
    return () => document.getElementById(id)?.remove();
  }, []);

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
    cancelRef.current = true;
    growRef.current   = false;
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

    for (let treeIdx = 0; treeIdx < trees.length; treeIdx++) {
      if (cancelRef.current) break;
      if (!trees[treeIdx]) continue;

      setCurTree(treeIdx);
      setTS(treeIdx, EMPTY_TS);
      await new Promise(r => setTimeout(r, 80));

      const steps = getSteps(treeIdx);
      for (let i = 0; i < steps.length; i++) {
        if (cancelRef.current) break;
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

  useEffect(() => {
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
                  <div style={{ position: "relative", display: "inline-flex", alignItems: "center" }}>
                    <select
                      className="ds-pill"
                      value={customDataset ? "custom" : builtinDataset}
                      disabled={disabled}
                      onChange={e => {
                        const val = e.target.value;
                        if (val === "upload") { fileInputRef.current?.click(); return; }
                        if (val === "custom") return;
                        switchToBuiltin(val);
                      }}
                      style={{ opacity: disabled ? 0.45 : 1, cursor: disabled ? "default" : "pointer" }}
                    >
                      <option value="heart">Built-in: Heart Disease (Binary classification)</option>
                      <option value="music">Built-in: Music Genres (Multiclass classification)</option>
                      <option value="salary">Built-in: Salary (Regression)</option>
                      {customDataset && <option value="custom">{customDataset.name}</option>}
                      <option value="upload">Upload CSV…</option>
                    </select>
                    {/* Chevron overlay */}
                    <span style={{
                      position: "absolute", right: 9, top: "50%",
                      transform: "translateY(-50%)",
                      pointerEvents: "none", color: C.dimmer, fontSize: 10, lineHeight: 1,
                    }}>▾</span>
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
        <div style={{ display: "flex", alignItems: "flex-end", gap: 12, flex: 1, flexWrap: "wrap" }}>
          {/* Max depth — always unlocked */}
          <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
            <label style={{ fontSize: 9, color: C.muted, fontWeight: 400 }}>Max depth</label>
            <input type="number" min={1} max={activeFeatures.length} value={maxDepthStr} disabled={growing}
              onChange={e => setMaxDepthStr(e.target.value)}
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
              onMouseEnter={() => setLockedParamTooltip("maxFeatures")}
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
              {/* Tooltip — sibling of dimmed content, inherits full opacity */}
              {lockedParamTooltip === "maxFeatures" && (
                <div style={{
                  position: "absolute", bottom: "calc(100% + 10px)", left: "50%",
                  transform: "translateX(-50%)",
                  width: 240, padding: "10px 13px", borderRadius: 10,
                  background: "#1a2235",
                  boxShadow: "0 8px 32px rgba(0,0,0,0.7), inset 0 0 0 1px rgba(255,255,255,0.12)",
                  fontSize: 10.5, color: C.text, lineHeight: 1.65,
                  fontWeight: 400, zIndex: 1000, pointerEvents: "none",
                  animation: "fadeInUp 0.14s ease-out",
                  whiteSpace: "normal",
                }}>
                  {mode === "decision-tree"
                    ? "A decision tree evaluates all features at each split — switch to Random Forest for feature subsampling"
                    : "Bagging typically considers all features at each split — switch to Random Forest to enable feature subsampling"}
                </div>
              )}
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
              onMouseEnter={() => setLockedParamTooltip("trees")}
              onMouseLeave={() => setLockedParamTooltip(null)}
            >
              {/* Dimmed control content */}
              <div style={{ display: "flex", flexDirection: "column", gap: 4, opacity: 0.45, pointerEvents: "none", userSelect: "none" }}>
                <label style={{ fontSize: 9, color: C.muted, fontWeight: 400, display: "flex", alignItems: "center", gap: 4 }}>
                  Trees <span style={{ fontSize: 9, lineHeight: 1 }}>🔒</span>
                </label>
                <input type="number" value={1} disabled style={{ ...inp, width: 58, cursor: "not-allowed" }} />
              </div>
              {/* Tooltip — sibling of dimmed content, inherits full opacity */}
              {lockedParamTooltip === "trees" && (
                <div style={{
                  position: "absolute", bottom: "calc(100% + 10px)", left: "50%",
                  transform: "translateX(-50%)",
                  width: 240, padding: "10px 13px", borderRadius: 10,
                  background: "#1a2235",
                  boxShadow: "0 8px 32px rgba(0,0,0,0.7), inset 0 0 0 1px rgba(255,255,255,0.12)",
                  fontSize: 10.5, color: C.text, lineHeight: 1.65,
                  fontWeight: 400, zIndex: 1000, pointerEvents: "none",
                  animation: "fadeInUp 0.14s ease-out",
                  whiteSpace: "normal",
                }}>
                  A decision tree is a single model — switch to an ensemble
                </div>
              )}
            </div>
          ) : (
            <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
              <label style={{ fontSize: 9, color: C.muted, fontWeight: 400 }}>Trees</label>
              <input type="number" min={1} max={100} value={nEstimatorsStr} disabled={growing}
                onChange={e => setNEstimatorsStr(e.target.value)}
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

        {/* Action buttons */}
        <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
          <button onClick={buildForest} disabled={!!buildProgress}
            onMouseEnter={e => { if (!buildProgress) { e.currentTarget.style.color = C.text; e.currentTarget.style.background = "rgba(255,255,255,0.06)"; } }}
            onMouseLeave={e => { e.currentTarget.style.color = C.dim; e.currentTarget.style.background = "none"; }}
            style={{
              padding: "7px 16px", borderRadius: 10, border: "none",
              background: "none", color: C.dim, fontSize: 11,
              fontFamily: "inherit", cursor: buildProgress ? "default" : "pointer", fontWeight: 500,
              transition: "color 0.15s ease-out, background 0.15s ease-out",
              opacity: buildProgress ? 0.4 : 1,
            }}>↻ Rebuild</button>
          <div style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 4 }}>
            <button
              disabled={!!buildProgress}
              onClick={() => {
                if (buildProgress) return;
                if (growing) { cancelRef.current = true; growRef.current = false; setGrowing(false); }
                else { setHintDismissed(true); autoGrowAll(); }
              }}
              onMouseEnter={e => { if (!buildProgress) e.currentTarget.style.transform = "scale(1.03)"; }}
              onMouseLeave={e => { e.currentTarget.style.transform = "scale(1)"; }}
              style={{
                padding: "8px 22px", borderRadius: 10, border: "none",
                background: buildProgress
                  ? "linear-gradient(135deg,#334155,#1e293b)"
                  : growing
                  ? "linear-gradient(135deg,#64748b,#475569)"
                  : `linear-gradient(135deg,${C.accent},#d97706)`,
                color: (buildProgress || growing) ? C.text : "#000", fontSize: 12, fontFamily: "inherit",
                cursor: buildProgress ? "default" : "pointer", fontWeight: 700,
                transition: "transform 0.15s ease-out, background 0.2s ease-out",
                animation: (buildProgress || growing) ? "none" : "growPulse 2.5s ease-in-out infinite",
              }}>
              {buildProgress ? "Building…" : growing ? "■ Stop" : "▶ Grow"}
            </button>
            {!growing && !buildProgress && (
              <div style={{ fontSize: 8.5, color: C.muted }}>
                or use ← → arrow keys
              </div>
            )}
          </div>
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
              <div ref={tabScrollRef} style={{
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
                          setTabTooltip({ x: r.left + r.width / 2, y: r.bottom + 6 });
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
          {nEstimators > 1 && (
            <button onClick={growAllInstant} disabled={growing || !!buildProgress}
              title="Instantly complete all trees without animation"
              onMouseEnter={e => { if (!growing && !buildProgress) { e.currentTarget.style.color = C.text; e.currentTarget.style.boxShadow = "inset 0 0 0 1px rgba(255,255,255,0.16)"; } }}
              onMouseLeave={e => { e.currentTarget.style.color = C.dim; e.currentTarget.style.boxShadow = "inset 0 0 0 1px rgba(255,255,255,0.08)"; }}
              style={{
                ...inp, flexShrink: 0,
                padding: "3px 10px", fontSize: 10,
                color: C.dim,
                cursor: (growing || buildProgress) ? "default" : "pointer",
                opacity: (growing || buildProgress) ? 0.4 : 1,
              }}>Complete all</button>
          )}

          {/* Step controls */}
          <button disabled={growing || !!buildProgress || ts.stepIdx <= -1} onClick={() => { setHintDismissed(true); goToStep(curTree, ts.stepIdx - 1); }}
            style={{ ...inp, cursor: (growing || buildProgress || ts.stepIdx <= -1) ? "default" : "pointer", padding: "3px 10px", opacity: (growing || buildProgress || ts.stepIdx <= -1) ? 0.3 : 1, fontSize: 11 }}>◀</button>
          <span style={{ fontSize: 10, color: C.muted, minWidth: 72, textAlign: "center", userSelect: "none" }}>
            {ts.stepIdx === -1 ? "ready" : `${ts.stepIdx + 1} / ${totalSteps}`}
          </span>
          <button disabled={growing || !!buildProgress || ts.stepIdx >= totalSteps - 1} onClick={() => { setHintDismissed(true); goToStep(curTree, ts.stepIdx + 1); }}
            style={{ ...inp, cursor: (growing || buildProgress || ts.stepIdx >= totalSteps - 1) ? "default" : "pointer", padding: "3px 10px", opacity: (growing || buildProgress || ts.stepIdx >= totalSteps - 1) ? 0.3 : 1, fontSize: 11 }}>▶</button>

          {/* Bootstrap info */}
          {curBootstrap && (
            <span style={{ fontSize: 9, color: C.muted, whiteSpace: "nowrap" }}>
              {curBootstrap.inBag}/{TOTAL_SAMPLES} in-bag · {curBootstrap.oob} OOB
              {curBootstrap.oobAccuracy > 0 ? ` · acc=${curBootstrap.oobAccuracy}` : ""}
            </span>
          )}

          {/* Reset */}
          <button onClick={() => setTS(curTree, EMPTY_TS)} disabled={growing}
            style={{ ...inp, cursor: "pointer", padding: "3px 10px", fontSize: 10, opacity: growing ? 0.3 : 1 }}>Reset</button>
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

      {/* SVG canvas */}
      <div ref={canvasRef}
        onMouseDown={onMouseDown} onMouseMove={onMouseMove}
        onMouseUp={onMouseUp} onMouseLeave={onMouseUp}
        style={{
          overflow: "hidden", cursor: cursorGrabbing ? "grabbing" : "grab",
          height: Math.min(500, svgH + 40), background: "#080c14",
          boxShadow: "0 2px 16px rgba(0,0,0,0.4)",
          position: "relative",
        }}>

        {/* Welcome overlay — shown on step 0, dismissed on first action or when any tree completes */}
        {!hintDismissed && completedTrees.length === 0 && !growing && !buildProgress && trees[0] && (
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
              <g key={node.id}>
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
            return <TreeNode key={node.id} node={node} show={show || ts.nodeId === node.id} phase={phase}
              pos={positions[node.id]} allClasses={classLabels}
              onPath={samplePath.has(node.id)} sampleActive={samplePath.size > 0}
              isRegression={activeTaskType === "regression"} />;
          })}
        </svg>

        {/* Floating zoom controls (bottom-right) */}
        <div style={{
          position: "absolute", bottom: 14, right: 14, zIndex: 10,
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

      {/* Feature pool */}
      <div style={{ padding: "12px 16px 6px" }}>
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

      {/* Calculations panel */}
      <div style={{ padding: "6px 16px 10px" }}>
        <div style={{ fontSize: 9, color: C.muted, fontWeight: 500, marginBottom: 6 }}>Calculations</div>
        <div style={{
          background: "#0c1018", borderRadius: 12,
          boxShadow: "0 2px 16px rgba(0,0,0,0.35), inset 0 0 0 1px rgba(255,255,255,0.04)",
          overflow: "hidden",
        }}>

          {/* Empty state */}
          {!currentNode && (
            <div style={{ padding: "18px 16px", color: C.dim, fontSize: 10.5 }}>
              Press <span style={{ color: C.muted }}>▶ Grow</span> or use <span style={{ color: C.muted }}>→</span> to begin…
            </div>
          )}

          {/* ── Leaf node ──────────────────────────────────────────────────────── */}
          {currentNode?.type === "leaf" && (() => {

            // ── Regression leaf ─────────────────────────────────────────────────
            if (currentNode.mean !== undefined) {
              return (
                <div style={{ padding: "16px 16px" }}>
                  <div style={{ fontSize: 9, color: C.muted, marginBottom: 10 }}>Leaf node · Prediction</div>
                  <div style={{ display: "flex", gap: 20, flexWrap: "wrap", marginBottom: 12 }}>
                    <div>
                      <div style={{ fontSize: 9, color: C.muted, marginBottom: 2 }}>Mean</div>
                      <div style={{ fontSize: 15, fontWeight: 700, color: C.blue, fontFamily: "'JetBrains Mono',monospace" }}>
                        {formatRegVal(currentNode.mean)}
                      </div>
                    </div>
                    <div>
                      <div style={{ fontSize: 9, color: C.muted, marginBottom: 2 }}>Variance</div>
                      <div style={{ fontSize: 12, color: C.text, fontFamily: "'JetBrains Mono',monospace" }}>
                        {currentNode.variance.toFixed(3)}
                      </div>
                    </div>
                    <div>
                      <div style={{ fontSize: 9, color: C.muted, marginBottom: 2 }}>n</div>
                      <div style={{ fontSize: 12, color: C.text, fontFamily: "'JetBrains Mono',monospace" }}>
                        {currentNode.samples}
                      </div>
                    </div>
                  </div>
                  <div style={{ fontSize: 9, color: C.muted }}>
                    Range: <span style={{ color: C.text }}>{formatRegVal(currentNode.min)} – {formatRegVal(currentNode.max)}</span>
                  </div>
                </div>
              );
            }

            // ── Classification leaf ──────────────────────────────────────────────
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
              <div style={{ padding: "16px 16px" }}>
                <div style={{ display: "flex", alignItems: "baseline", gap: 8, marginBottom: 12 }}>
                  <span style={{ fontSize: 9, color: C.muted }}>Leaf node · Prediction</span>
                </div>
                <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 14 }}>
                  <div style={{
                    fontSize: 15, fontWeight: 700, color: predColor,
                    fontFamily: "'JetBrains Mono',monospace",
                  }}>{currentNode.prediction}</div>
                  <div style={{ fontSize: 9, color: C.muted }}>
                    Gini <span style={{ color: C.text, fontFamily: "'JetBrains Mono',monospace" }}>{currentNode.impurity.toFixed(4)}</span>
                    <span style={{ marginLeft: 8 }}>n = <span style={{ color: C.text, fontFamily: "'JetBrains Mono',monospace" }}>{currentNode.samples}</span></span>
                  </div>
                </div>

                {/* Stacked distribution bar */}
                <div style={{ height: 8, borderRadius: 4, overflow: "hidden", background: "rgba(255,255,255,0.05)", marginBottom: 8, display: "flex" }}>
                  {barSegs.map(s => (
                    <div key={s.cls} style={{ width: `${s.pct}%`, background: s.color, transition: "width .3s" }} />
                  ))}
                </div>

                {/* Class count legend */}
                <div style={{ display: "flex", flexWrap: "wrap", gap: "4px 14px" }}>
                  {sortedCounts.map(([cls, cnt]) => (
                    <div key={cls} style={{ display: "flex", alignItems: "center", gap: 5, fontSize: 9 }}>
                      <div style={{ width: 7, height: 7, borderRadius: 2, background: classColor(cls, classLabels), flexShrink: 0 }} />
                      <span style={{ color: C.dim }}>{cls}</span>
                      <span style={{ color: C.text, fontFamily: "'JetBrains Mono',monospace" }}>{cnt}</span>
                    </div>
                  ))}
                </div>
              </div>
            );
          })()}

          {/* ── Split node ─────────────────────────────────────────────────────── */}
          {currentNode?.type === "split" && (() => {
            const candidateEvals = (currentNode.allFeatureEvals ?? [])
              .filter(ev => currentNode.candidateIndices.includes(ev.featureIndex))
              .sort((a, b) => a.gini - b.gini); // sorted best-first for bar chart

            // Gini range for bar normalisation — use the worst gini as max bar width
            const maxGini = Math.max(...candidateEvals.map(e => e.gini), 0.001);

            return (<>
              {/* ① Node header — always visible */}
              <div style={{ padding: "14px 16px 12px" }}>
                <div style={{ fontSize: 10, color: C.text, fontWeight: 600, marginBottom: 2 }}>
                  Depth {currentNode.depth} · {currentNode.samples} samples
                </div>
                {ts.phase === 0 && (
                  <div style={{ display: "flex", alignItems: "center", gap: 8, marginTop: 8 }}>
                    <div style={{
                      display: "flex", gap: 3, alignItems: "center",
                    }}>
                      {[0,1,2].map(i => (
                        <div key={i} style={{
                          width: 5, height: 5, borderRadius: "50%",
                          background: C.accent, opacity: 0.3 + i * 0.3,
                          animation: `growPulse ${1 + i * 0.2}s ease-in-out infinite`,
                        }} />
                      ))}
                    </div>
                    <span style={{ fontSize: 9.5, color: C.dim }}>
                      Sampling {currentNode.candidateIndices?.length ?? "?"} of {activeFeatures.length} features…
                    </span>
                  </div>
                )}
                {ts.phase >= 1 && (
                  <div style={{ marginTop: 4, fontSize: 9, color: C.dim, lineHeight: 1.6 }}>
                    Evaluating{" "}
                    <span style={{ color: `${C.accent}cc` }}>
                      {currentNode.candidateIndices.map(i => activeFeatures[i]).join(", ")}
                    </span>
                  </div>
                )}
              </div>

              {/* ② Gini bar chart — phase 1+ */}
              {ts.phase >= 1 && (
                <div style={{
                  borderTop: "1px solid rgba(255,255,255,0.05)",
                  padding: "12px 16px",
                }}>
                  <div style={{ fontSize: 9, color: C.dim, marginBottom: 8 }}>{activeTaskType === "regression" ? "MSE" : "Gini"} per candidate <span style={{ color: C.muted }}>(shorter = better)</span></div>
                  <div style={{ display: "flex", flexDirection: "column", gap: 5 }}>
                    {candidateEvals.map((ev, j) => {
                      const isChosen = ts.phase >= 2 && ev.featureIndex === currentNode.featureIndex;
                      const barPct = maxGini > 0 ? (ev.gini / maxGini) * 100 : 50;
                      const rowColor = isChosen ? C.green : C.dim;
                      return (
                        <div key={ev.featureIndex} style={{
                          display: "flex", alignItems: "center", gap: 8,
                          opacity: 0,
                          animation: `taxFadeIn 0.25s ease ${j * 0.06}s forwards`,
                        }}>
                          {/* Feature name */}
                          <div style={{
                            width: 110, fontSize: 9, color: rowColor, fontWeight: isChosen ? 600 : 400,
                            overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap",
                            flexShrink: 0, transition: "color .2s",
                          }}>
                            {activeFeatures[ev.featureIndex]}
                          </div>
                          {/* Bar */}
                          <div style={{ flex: 1, height: 5, background: "rgba(255,255,255,0.05)", borderRadius: 3, overflow: "hidden" }}>
                            <div style={{
                              height: "100%",
                              width: `${barPct}%`,
                              background: isChosen ? C.green : `${C.accent}88`,
                              borderRadius: 3,
                              transition: "width 0.4s ease-out, background 0.2s",
                            }} />
                          </div>
                          {/* Gini value */}
                          <div style={{
                            fontSize: 9, color: rowColor, fontFamily: "'JetBrains Mono',monospace",
                            flexShrink: 0, width: 46, textAlign: "right",
                            transition: "color .2s",
                          }}>
                            {ev.gini.toFixed(4)}
                            {isChosen && <span style={{ marginLeft: 4, fontSize: 8 }}>✓</span>}
                          </div>
                        </div>
                      );
                    })}
                  </div>
                </div>
              )}

              {/* ③ Global-best warning — phase 2 only, when subset missed the true best */}
              {ts.phase >= 2 && currentNode.globalBestIdx !== currentNode.featureIndex && (
                <div style={{
                  borderTop: "1px solid rgba(255,255,255,0.05)",
                  margin: "0 16px",
                  padding: "10px 12px",
                  borderRadius: 8,
                  background: `${C.accent}0a`,
                  borderLeft: `3px solid ${C.accent}88`,
                  marginBottom: 10,
                  fontSize: 9, color: C.dim, lineHeight: 1.65,
                }}>
                  <span style={{ color: `${C.accent}cc`, fontWeight: 600 }}>Subset missed global best</span>
                  <br />
                  <span style={{ fontFamily: "'JetBrains Mono',monospace", color: C.text }}>
                    {activeFeatures[currentNode.globalBestIdx]}
                  </span>{" "}
                  had {activeTaskType === "regression" ? "MSE" : "G"}={currentNode.globalBestGini.toFixed(4)} but wasn't sampled into the subset
                </div>
              )}

              {/* ④ Split result card — phase 2 */}
              {ts.phase >= 2 && (
                <div style={{
                  borderTop: "1px solid rgba(255,255,255,0.05)",
                  padding: "12px 16px",
                }}>
                  <div style={{ fontSize: 9, color: C.dim, marginBottom: 6 }}>Split decision</div>
                  <div style={{
                    display: "flex", alignItems: "baseline", gap: 6,
                    padding: "10px 12px", borderRadius: 8,
                    background: `${C.green}0f`,
                    borderLeft: `3px solid ${C.green}88`,
                  }}>
                    <span style={{
                      fontSize: 12, fontWeight: 700, color: C.green,
                      fontFamily: "'JetBrains Mono',monospace",
                    }}>
                      {currentNode.featureName}
                    </span>
                    <span style={{ fontSize: 11, color: C.dim, fontFamily: "'JetBrains Mono',monospace" }}>
                      ≤ {currentNode.threshold}
                    </span>
                    <span style={{ fontSize: 9, color: C.muted, marginLeft: 4 }}>
                      {activeTaskType === "regression" ? "MSE" : "G"}={currentNode.gini.toFixed(4)}
                    </span>
                  </div>
                </div>
              )}
            </>);
          })()}
        </div>

        {/* ↓ Back to prediction — appears when a sample is selected */}
        {selectedSample && (
          <button
            onClick={() => predSectionRef.current?.scrollIntoView({ behavior: "smooth", block: "start" })}
            style={{
              position: "absolute", bottom: 10, right: 10,
              padding: "5px 13px", borderRadius: 20,
              background: "rgba(10,14,23,0.78)", backdropFilter: "blur(6px)",
              border: `1px solid rgba(255,255,255,0.1)`,
              color: C.muted, fontSize: 10, fontWeight: 600,
              cursor: "pointer", fontFamily: "inherit",
              boxShadow: "0 2px 12px rgba(0,0,0,0.4)",
              transition: "color 0.15s, border-color 0.15s",
              zIndex: 6,
            }}
            onMouseEnter={e => { e.currentTarget.style.color = C.text; e.currentTarget.style.borderColor = "rgba(255,255,255,0.22)"; }}
            onMouseLeave={e => { e.currentTarget.style.color = C.muted; e.currentTarget.style.borderColor = "rgba(255,255,255,0.1)"; }}
          >
            ↓ Back to prediction
          </button>
        )}
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

            {/* ── Prominent sample selector (Spotlight style) ──────────────── */}
            <div style={{
              position: "relative",
              display: "flex",
              alignItems: "center",
              marginBottom: 20,
              borderRadius: 16,
              background: "rgba(255,255,255,0.04)",
              boxShadow: "0 8px 32px rgba(0,0,0,0.4), 0 0 0 1px rgba(255,255,255,0.08), inset 0 1px 0 rgba(255,255,255,0.06)",
              overflow: "hidden",
            }}>
              {/* Search icon */}
              <div style={{ padding: "0 14px 0 18px", color: C.dim, fontSize: 16, flexShrink: 0, pointerEvents: "none" }}>
                ⌕
              </div>
              <select
                value={safeSampleIdx ?? ""}
                onChange={e => setSelectedSampleIdx(e.target.value === "" ? null : +e.target.value)}
                style={{
                  flex: 1, minWidth: 0, background: "transparent", border: "none",
                  color: safeSampleIdx !== null ? C.text : C.muted,
                  fontSize: 13, fontFamily: "'JetBrains Mono',monospace",
                  padding: "15px 8px 15px 0", outline: "none",
                  cursor: "pointer", appearance: "none", WebkitAppearance: "none",
                }}
              >
                <option value="">Pick a sample to predict…</option>
                {activeData.map((row, i) => (
                  <option key={i} value={i}>{formatSampleLabel(row, activeFeatures, i)}</option>
                ))}
              </select>
              {/* Random pill */}
              <button
                onClick={() => setSelectedSampleIdx(Math.floor(Math.random() * activeData.length))}
                style={{
                  flexShrink: 0, margin: "8px 10px 8px 4px",
                  padding: "6px 16px", borderRadius: 20, border: "none",
                  background: "rgba(255,255,255,0.07)", color: C.muted, fontSize: 11,
                  cursor: "pointer", fontFamily: "inherit", fontWeight: 600,
                  transition: "color .15s, background .15s",
                  boxShadow: "inset 0 0 0 1px rgba(255,255,255,0.1)",
                }}
                onMouseEnter={e => { e.currentTarget.style.color = C.text; e.currentTarget.style.background = "rgba(255,255,255,0.13)"; }}
                onMouseLeave={e => { e.currentTarget.style.color = C.muted; e.currentTarget.style.background = "rgba(255,255,255,0.07)"; }}
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

            {/* ── Per-tree vote cards ─────────────────────────────────────────── */}
            {selectedSample && (
              <>
                <div style={{ display: "flex", gap: 5, flexWrap: "wrap", alignItems: "center", marginBottom: 14 }}>
                  {trees.map((_, i) => {
                    const spt     = sampleTreePreds.find(t => t.idx === i);
                    const hasPred = spt !== undefined;
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
                          }}>
                          <div style={{ fontSize: 7, color: isActive ? C.text : C.dim }}>T{i + 1}</div>
                          <div style={{ fontSize: 9, fontWeight: 700, color: hasPred ? C.blue : C.dimmer, whiteSpace: "nowrap" }}>
                            {predStr}
                          </div>
                        </button>
                      );
                    }

                    const pred    = spt?.samplePred;
                    const color   = pred ? classColor(pred, classLabels) : C.dimmer;
                    const abbrev  = pred && pred.length > 9 ? pred.slice(0, 8) + "…" : pred;
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
                        }}>
                        <div style={{ fontSize: 7, color: isActive ? C.text : C.dim }}>T{i + 1}</div>
                        <div style={{ fontSize: 9, fontWeight: 700, color: pred ? color : C.dimmer, whiteSpace: "nowrap" }}>
                          {pred ? abbrev : "…"}
                        </div>
                      </button>
                    );
                  })}
                </div>

                {/* ── Ensemble result — prominent card ────────────────────────── */}
                {activeTaskType === "regression" && sampleMean !== null && (() => {
                  const err = sampleTrueLabel !== null ? Math.abs(sampleMean - sampleTrueLabel) : null;
                  const relErr = err !== null && Math.abs(sampleTrueLabel) > 0 ? err / Math.abs(sampleTrueLabel) : null;
                  const errColor = relErr !== null ? (relErr < 0.1 ? C.green : relErr < 0.25 ? C.orange : C.red) : C.muted;
                  return (
                    <div style={{
                      marginBottom: 16, padding: "18px 20px",
                      borderRadius: 14,
                      background: `linear-gradient(135deg, ${C.blue}12, rgba(10,14,23,0.6))`,
                      boxShadow: `0 0 0 1.5px ${C.blue}55, 0 4px 24px ${C.blue}18, inset 0 1px 0 ${C.blue}22`,
                    }}>
                      <div style={{ fontSize: 9, color: C.dim, fontWeight: 500, marginBottom: 6, letterSpacing: "0.06em", textTransform: "uppercase" }}>
                        Ensemble prediction · {completedTrees.length} tree{completedTrees.length !== 1 ? "s" : ""}
                      </div>
                      <div style={{ fontSize: 28, fontWeight: 800, color: C.blue, fontFamily: "'JetBrains Mono',monospace", lineHeight: 1, marginBottom: 10 }}>
                        {formatRegVal(sampleMean)}
                      </div>
                      {sampleTrueLabel !== null && (
                        <div style={{ display: "flex", gap: 16, flexWrap: "wrap", alignItems: "center" }}>
                          <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
                            <span style={{ fontSize: 9, color: C.muted, textTransform: "uppercase", letterSpacing: "0.05em" }}>True</span>
                            <span style={{ fontSize: 14, fontWeight: 700, color: C.text, fontFamily: "'JetBrains Mono',monospace" }}>{formatRegVal(sampleTrueLabel)}</span>
                          </div>
                          <div style={{ width: 1, height: 16, background: "rgba(255,255,255,0.1)" }} />
                          <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
                            <span style={{ fontSize: 9, color: C.muted, textTransform: "uppercase", letterSpacing: "0.05em" }}>Error</span>
                            <span style={{ fontSize: 14, fontWeight: 700, color: errColor, fontFamily: "'JetBrains Mono',monospace" }}>{formatRegVal(err)}</span>
                          </div>
                        </div>
                      )}
                    </div>
                  );
                })()}

                {/* ── Classification ensemble result — prominent card ───────────── */}
                {activeTaskType === "classification" && sampleMajority && (() => {
                  const majColor = classColor(sampleMajority, classLabels);
                  return (
                    <div style={{
                      marginBottom: 16, padding: "18px 20px",
                      borderRadius: 14,
                      background: `linear-gradient(135deg, ${majColor}12, rgba(10,14,23,0.6))`,
                      boxShadow: `0 0 0 1.5px ${majColor}55, 0 4px 24px ${majColor}18, inset 0 1px 0 ${majColor}22`,
                    }}>
                      <div style={{ fontSize: 9, color: C.dim, fontWeight: 500, marginBottom: 6, letterSpacing: "0.06em", textTransform: "uppercase" }}>
                        Ensemble prediction · {completedTrees.length} tree{completedTrees.length !== 1 ? "s" : ""}
                      </div>
                      <div style={{ fontSize: 26, fontWeight: 800, color: majColor, lineHeight: 1, marginBottom: 10 }}>
                        {sampleMajority}
                      </div>
                      {/* Vote bar */}
                      <div style={{ marginBottom: 10 }}>
                        <div style={{ height: 6, borderRadius: 3, overflow: "hidden", background: "rgba(0,0,0,0.3)", display: "flex", marginBottom: 5 }}>
                          {classLabels.map(cls => {
                            const pct = ((sampleVotesPerClass[cls] ?? 0) / completedTrees.length) * 100;
                            return pct > 0
                              ? <div key={cls} style={{ width: `${pct}%`, background: classColor(cls, classLabels), transition: "width .3s" }} />
                              : null;
                          })}
                        </div>
                        <div style={{ display: "flex", gap: 10, flexWrap: "wrap" }}>
                          {classLabels.map(cls => {
                            const n = sampleVotesPerClass[cls] ?? 0;
                            if (n === 0) return null;
                            return (
                              <span key={cls} style={{ fontSize: 9, color: C.muted }}>
                                <strong style={{ color: classColor(cls, classLabels), fontFamily: "'JetBrains Mono',monospace" }}>{n}</strong>
                                <span style={{ color: C.dimmer }}>/{completedTrees.length} </span>
                                <span style={{ color: classColor(cls, classLabels) }}>{cls}</span>
                              </span>
                            );
                          })}
                        </div>
                      </div>
                      {sampleTrueLabel !== null && (
                        <div style={{ display: "flex", gap: 14, flexWrap: "wrap", alignItems: "center", paddingTop: 10, borderTop: "1px solid rgba(255,255,255,0.06)" }}>
                          <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
                            <span style={{ fontSize: 9, color: C.muted, textTransform: "uppercase", letterSpacing: "0.05em" }}>True</span>
                            <span style={{ fontSize: 14, fontWeight: 700, color: classColor(sampleTrueLabel, classLabels) }}>{sampleTrueLabel}</span>
                          </div>
                          <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
                            <span style={{ fontSize: 14, fontWeight: 800, color: sampleCorrect ? C.green : C.red }}>
                              {sampleCorrect ? "✓ Correct" : "✗ Wrong"}
                            </span>
                          </div>
                        </div>
                      )}
                    </div>
                  );
                })()}
              </>
            )}

            {/* ── Metrics cards ───────────────────────────────────────────────── */}
            {activeTaskType === "classification" && (forestAccuracy !== null || avgOobAccuracy !== null) && (
              <div style={{ display: "flex", gap: 10, marginTop: selectedSample ? 4 : 0 }}>
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
                        onMouseEnter={() => setOobTooltipVisible(true)}
                        onMouseLeave={() => setOobTooltipVisible(false)}>ⓘ</span>
                    </div>
                    <div style={{ fontSize: 9, color: C.muted, marginBottom: 2 }}>Accuracy</div>
                    <div style={{ fontSize: 20, fontWeight: 800, color: C.green, fontFamily: "'JetBrains Mono',monospace" }}>{avgOobAccuracy}%</div>
                    {oobTooltipVisible && (
                      <div style={{ position: "absolute", bottom: "calc(100% + 6px)", left: 0, width: 240, background: "#141b2d", borderRadius: 9, padding: "9px 12px", fontSize: 9.5, color: C.dim, lineHeight: 1.6, boxShadow: "0 8px 32px rgba(0,0,0,0.6), inset 0 0 0 1px rgba(255,255,255,0.08)", zIndex: 20, pointerEvents: "none" }}>
                        Each tree is tested on the ~37% of samples not used in its bootstrap training set.
                        This gives an unbiased estimate of generalization error with no held-out test set needed.
                      </div>
                    )}
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
                        onMouseEnter={() => setOobTooltipVisible(true)}
                        onMouseLeave={() => setOobTooltipVisible(false)}>ⓘ</span>
                    </div>
                    <div style={{ fontSize: 9, color: C.muted, marginBottom: 2 }}>R²</div>
                    <div style={{ fontSize: 20, fontWeight: 800, color: C.green, fontFamily: "'JetBrains Mono',monospace" }}>{avgOobR2}</div>
                    <div style={{ fontSize: 9, color: C.muted, marginTop: 6 }}>RMSE <span style={{ color: C.green, fontFamily: "'JetBrains Mono',monospace", fontWeight: 700 }}>{avgOobRMSE}</span></div>
                    {oobTooltipVisible && (
                      <div style={{ position: "absolute", bottom: "calc(100% + 6px)", left: 0, width: 260, background: "#141b2d", borderRadius: 9, padding: "9px 12px", fontSize: 9.5, color: C.dim, lineHeight: 1.6, boxShadow: "0 8px 32px rgba(0,0,0,0.6), inset 0 0 0 1px rgba(255,255,255,0.08)", zIndex: 20, pointerEvents: "none" }}>
                        R² measures proportion of variance explained. 1.0 is perfect, 0.0 is no better than predicting the mean.
                        OOB estimates this on the ~37% of samples held out from each tree's bootstrap.
                      </div>
                    )}
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
          left: tabTooltip.x,
          top: tabTooltip.y,
          transform: "translateX(-50%)",
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
    </div>
  );
}
