import { useState, useEffect, useRef, useCallback, useMemo } from "react";
import { useLocation } from "react-router-dom";
import Papa from "papaparse";
import { C } from "./theme";
import GlobalHeader from "./GlobalHeader";
import { heartData, heartMeta } from "./data/heartDisease";
import { musicData, musicMeta } from "./data/musicData";
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
  return (
    <g style={{ opacity: dimmed ? 0.15 : 1, transition: "opacity .3s" }}>
      <line x1={p1.x} y1={p1.y + 27} x2={p2.x} y2={p2.y - 24}
        stroke={highlighted ? PATH_COLOR : C.edge}
        strokeWidth={highlighted ? 2.4 : 1.4}
        strokeDasharray={highlighted ? undefined : "4 2"}
        style={{ transition: "stroke .25s, stroke-width .25s" }}
      />
      <text x={(p1.x + p2.x) / 2} y={(p1.y + 27 + p2.y - 24) / 2 - 4}
        textAnchor="middle" fill={highlighted ? PATH_COLOR : C.dim}
        fontSize={8} fontFamily="'JetBrains Mono',monospace">{label}</text>
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
          <text x={x} y={y + 16} textAnchor="middle" fill={C.dimmer} fontSize={7}
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
  if (sampleMode === "1000" && rows.length > 1000) {
    if (taskType === "classification") {
      rows = stratifiedSample(rows, targetCol, 1000);
    } else {
      // Random sample for regression (no discrete classes to stratify on)
      for (let i = rows.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [rows[i], rows[j]] = [rows[j], rows[i]];
      }
      rows = rows.slice(0, 1000);
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

const SAMPLE_OPTIONS = [
  { key: "1000", label: "Random sample: 1,000 rows", sub: "Stratified by target class" },
  { key: "all",  label: "Use all rows",              sub: "May be slow for large datasets" },
];

// ─── CSV Modal ─────────────────────────────────────────────────────────────────
function DataModal({ modal, onUpdate, onConfirm, onCancel }) {
  const { fileName, rawRows, headers, naStats, selectedTarget, naStrategy, sampleMode, selectedColumns, warning, taskType = "classification" } = modal;
  const includedCols = selectedColumns ?? headers;
  const isLarge = rawRows.length > 1000;

  const inp = {
    padding: "5px 8px", borderRadius: 8, background: "#161c2a",
    border: "none", color: C.text, fontSize: 11,
    fontFamily: "'JetBrains Mono',monospace", outline: "none",
    boxShadow: "inset 0 0 0 1px rgba(255,255,255,0.08)",
  };

  const toggleCol = (h) => {
    if (h === selectedTarget) return;
    const next = includedCols.includes(h)
      ? includedCols.filter(c => c !== h)
      : [...includedCols, h];
    if (!next.includes(selectedTarget)) next.push(selectedTarget);
    onUpdate({ selectedColumns: next });
  };

  const handleConfirm = () => {
    const result = processCSVData(rawRows, headers, selectedTarget, naStrategy, sampleMode ?? "1000", includedCols, taskType);
    if (!result) return;
    onConfirm(result.data, result.features, result.targetCol,
              fileName.replace(".csv", ""), result.totalRows, result.sampledRows, result.classLabels, taskType);
  };

  const featureCols  = headers.filter(h => h !== selectedTarget);
  const checkedCount = featureCols.filter(h => includedCols.includes(h)).length;

  return (
    <div style={{
      position: "fixed", inset: 0, background: "rgba(0,0,0,0.75)", zIndex: 100,
      display: "flex", alignItems: "center", justifyContent: "center",
      backdropFilter: "blur(4px)",
    }}>
      <div style={{
        background: C.panel, borderRadius: 16,
        boxShadow: "0 24px 64px rgba(0,0,0,0.7), inset 0 0 0 1px rgba(255,255,255,0.07)",
        padding: 24, maxWidth: 480, width: "90vw", fontFamily: "'JetBrains Mono',monospace",
        color: C.text, maxHeight: "88vh", overflowY: "auto",
      }}>
        <div style={{ fontSize: 13, fontWeight: 700, color: C.accent, marginBottom: 14 }}>
          Configure dataset
        </div>

        {/* File summary */}
        <div style={{ fontSize: 10, color: C.dim, marginBottom: 16, lineHeight: 1.8 }}>
          <div>📄 <strong style={{ color: C.text }}>{fileName}</strong></div>
          <div>{rawRows.length.toLocaleString()} rows · {headers.length} columns</div>
          {naStats.total > 0 && (
            <div style={{ color: C.orange }}>⚠ {naStats.total} missing values found</div>
          )}
          {warning && <div style={{ color: C.red, marginTop: 4 }}>⚠ {warning}</div>}
        </div>

        {/* Task type */}
        <div style={{ marginBottom: 16 }}>
          <label style={{ fontSize: 9, color: C.dim, fontWeight: 400, display: "block", marginBottom: 6 }}>
            Task type
          </label>
          <div style={{ display: "flex", gap: 8, marginBottom: 5 }}>
            {["classification", "regression"].map(t => (
              <button key={t} onClick={() => onUpdate({ taskType: t })} style={{
                flex: 1, padding: "6px 10px", borderRadius: 9, fontSize: 10, cursor: "pointer",
                fontFamily: "inherit", fontWeight: taskType === t ? 600 : 400,
                background: taskType === t ? `${C.accent}22` : "rgba(255,255,255,0.04)",
                border: "none",
                boxShadow: taskType === t ? `inset 0 0 0 1px ${C.accent}66` : "inset 0 0 0 1px rgba(255,255,255,0.08)",
                color: taskType === t ? C.accent : C.dim,
                transition: "all 0.15s ease-out",
              }}>
                {t === "classification" ? "Classification" : "Regression"}
              </button>
            ))}
          </div>
          <div style={{ fontSize: 9, color: C.dimmer }}>
            {taskType === "regression"
              ? "Predicts a continuous numeric value · MSE split criterion"
              : "Predicts a class label · Gini impurity split criterion"}
          </div>
        </div>

        {/* Target column */}
        <div style={{ marginBottom: 16 }}>
          <label style={{ fontSize: 9, color: C.dim, fontWeight: 400, display: "block", marginBottom: 5 }}>
            Target column
          </label>
          <select value={selectedTarget}
            onChange={e => {
              const newTarget = e.target.value;
              const detected  = detectTaskType(rawRows, newTarget);
              onUpdate({ selectedTarget: newTarget, selectedColumns: headers, taskType: detected });
            }}
            style={{ ...inp, width: "100%", cursor: "pointer" }}>
            {headers.map(h => <option key={h} value={h}>{h}</option>)}
          </select>
          <div style={{ fontSize: 9, color: C.dimmer, marginTop: 3 }}>
            {taskType === "regression"
              ? "Numeric target — predicts continuous values"
              : "Sorted unique values mapped to class labels"}
          </div>
        </div>

        {/* Feature column selection */}
        <div style={{ marginBottom: 16 }}>
          <div style={{ display: "flex", alignItems: "baseline", gap: 8, marginBottom: 7 }}>
            <label style={{ fontSize: 9, color: C.dim, fontWeight: 400 }}>
              Select features to include
            </label>
            <span style={{ fontSize: 8.5, color: C.dimmer }}>
              {checkedCount}/{featureCols.length} selected
            </span>
          </div>
          <div style={{
            maxHeight: 168, overflowY: "auto", display: "flex", flexDirection: "column", gap: 1,
            background: "#0c1018", borderRadius: 10,
            boxShadow: "inset 0 0 0 1px rgba(255,255,255,0.06)",
            padding: "4px 0",
            scrollbarWidth: "thin", scrollbarColor: `${C.border} transparent`,
          }}>
            {/* Target row — always checked, disabled */}
            <label style={{
              display: "flex", alignItems: "center", gap: 8,
              padding: "5px 12px", cursor: "default", opacity: 0.45,
            }}>
              <input type="checkbox" checked readOnly
                style={{ accentColor: C.accent, cursor: "default", flexShrink: 0 }} />
              <span style={{ fontSize: 10, color: C.dim, flex: 1, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                {selectedTarget}
              </span>
              <span style={{ fontSize: 8, color: C.dimmer, flexShrink: 0 }}>target</span>
            </label>
            {featureCols.map(h => {
              const checked = includedCols.includes(h);
              return (
                <label key={h} style={{
                  display: "flex", alignItems: "center", gap: 8,
                  padding: "5px 12px", cursor: "pointer",
                  transition: "background 0.12s",
                }}
                onMouseEnter={e => { e.currentTarget.style.background = "rgba(255,255,255,0.03)"; }}
                onMouseLeave={e => { e.currentTarget.style.background = "transparent"; }}
                >
                  <input type="checkbox" checked={checked} onChange={() => toggleCol(h)}
                    style={{ accentColor: C.accent, cursor: "pointer", flexShrink: 0 }} />
                  <span style={{
                    fontSize: 10, color: checked ? C.text : C.dim,
                    flex: 1, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap",
                    transition: "color 0.12s",
                  }}>
                    {h}
                  </span>
                </label>
              );
            })}
          </div>
        </div>

        {/* NA strategy */}
        {naStats.total > 0 && (
          <div style={{ marginBottom: 16 }}>
            <label style={{ fontSize: 9, color: C.dim, fontWeight: 400, display: "block", marginBottom: 6 }}>
              Handle missing values
            </label>
            <div style={{ display: "flex", gap: 8 }}>
              {["drop", "median"].map(s => (
                <button key={s} onClick={() => onUpdate({ naStrategy: s })} style={{
                  flex: 1, padding: "6px 10px", borderRadius: 9, fontSize: 10, cursor: "pointer",
                  fontFamily: "inherit", fontWeight: naStrategy === s ? 600 : 400,
                  background: naStrategy === s ? `${C.accent}22` : "rgba(255,255,255,0.04)",
                  border: "none",
                  boxShadow: naStrategy === s ? `inset 0 0 0 1px ${C.accent}66` : "inset 0 0 0 1px rgba(255,255,255,0.08)",
                  color: naStrategy === s ? C.accent : C.dim,
                  transition: "all 0.15s ease-out",
                }}>
                  {s === "drop" ? "Drop rows with NA" : "Fill with median"}
                </button>
              ))}
            </div>
          </div>
        )}

        {/* Sampling — only shown for datasets over 1,000 rows */}
        {isLarge && (
          <div style={{ marginBottom: 16 }}>
            <label style={{ fontSize: 9, color: C.dim, fontWeight: 400, display: "block", marginBottom: 6 }}>
              Sampling
            </label>
            <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
              {SAMPLE_OPTIONS.map(opt => {
                const active = (sampleMode ?? "1000") === opt.key;
                return (
                  <button key={opt.key} onClick={() => onUpdate({ sampleMode: opt.key })} style={{
                    padding: "8px 12px", borderRadius: 9, fontSize: 10, cursor: "pointer",
                    fontFamily: "inherit", textAlign: "left", display: "flex", justifyContent: "space-between", alignItems: "center",
                    background: active ? `${C.accent}22` : "rgba(255,255,255,0.04)",
                    border: "none",
                    boxShadow: active ? `inset 0 0 0 1px ${C.accent}66` : "inset 0 0 0 1px rgba(255,255,255,0.08)",
                    color: active ? C.accent : C.dim,
                    fontWeight: active ? 600 : 400,
                    transition: "all 0.15s ease-out",
                  }}>
                    <span>{opt.label}</span>
                    <span style={{ fontSize: 8, opacity: 0.6 }}>{opt.sub}</span>
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
          <button onClick={onCancel}
            onMouseEnter={e => { e.currentTarget.style.color = C.text; e.currentTarget.style.background = "rgba(255,255,255,0.06)"; }}
            onMouseLeave={e => { e.currentTarget.style.color = C.dim; e.currentTarget.style.background = "none"; }}
            style={{
              flex: 1, padding: "9px", borderRadius: 10, border: "none",
              background: "none", color: C.dim, fontSize: 11, fontFamily: "inherit", cursor: "pointer",
              transition: "color 0.15s ease-out, background 0.15s ease-out",
            }}>Cancel</button>
          <button onClick={handleConfirm} style={{
            flex: 2, padding: "9px", borderRadius: 10, border: "none",
            background: `linear-gradient(135deg,${C.accent},#d97706)`,
            color: "#000", fontSize: 11, fontFamily: "inherit", cursor: "pointer", fontWeight: 700,
            transition: "transform 0.15s ease-out",
          }}
          onMouseEnter={e => { e.currentTarget.style.transform = "scale(1.02)"; }}
          onMouseLeave={e => { e.currentTarget.style.transform = "scale(1)"; }}
          >Confirm &amp; Build</button>
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
  const [builtinDataset, setBuiltinDataset] = useState("heart"); // "heart" | "music"
  const [customDataset, setCustomDataset] = useState(null);
  const fileInputRef = useRef(null);
  const [csvModal, setCsvModal]           = useState(null);
  const [dragOver, setDragOver]           = useState(false);
  const [selectedSampleIdx, setSelectedSampleIdx] = useState(null);
  const [oobTooltipVisible, setOobTooltipVisible] = useState(false);

  const builtinMeta     = builtinDataset === "music" ? musicMeta : heartMeta;
  const builtinData     = builtinDataset === "music" ? musicData : heartData;
  const activeData      = customDataset?.data        ?? builtinData;
  const activeFeatures  = customDataset?.features    ?? builtinMeta.features;
  const activeTargetCol = customDataset?.targetCol   ?? builtinMeta.targetCol;
  const classLabels     = customDataset?.classLabels ?? builtinMeta.targetLabels;
  const activeTaskType  = customDataset?.taskType    ?? "classification";

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
    const d = key === "music" ? musicData : heartData;
    const m = key === "music" ? musicMeta : heartMeta;
    buildForestWithData(d, m.features, m.targetCol, "classification");
  }, [buildForestWithData]);

  useEffect(() => {
    const pending = location.state?.pendingCSV;
    if (pending) {
      // A CSV was dropped on the landing page — open the DataModal automatically
      const { data: rawRows, meta } = Papa.parse(pending.content, { header: true, skipEmptyLines: true });
      if (rawRows.length) {
        const headers    = meta.fields;
        const naStats    = detectNAs(rawRows, headers);
        const isLarge    = rawRows.length > 1000;
        const initTarget = headers[headers.length - 1];
        const modal = {
          fileName: pending.name, rawRows, headers, naStats,
          selectedTarget: initTarget,
          naStrategy: "drop", sampleMode: isLarge ? "1000" : "all",
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
        await new Promise(r => setTimeout(r, 400 / speed));
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

  // ── File drop handling ─────────────────────────────────────────────────────
  const openFile = useCallback((file) => {
    if (!file || !file.name.toLowerCase().endsWith(".csv")) return;
    const reader = new FileReader();
    reader.onload = (e) => {
      const { data: rawRows, meta } = Papa.parse(e.target.result, { header: true, skipEmptyLines: true });
      if (!rawRows.length) return;
      const headers    = meta.fields;
      const naStats    = detectNAs(rawRows, headers);
      const isLarge    = rawRows.length > 1000;
      const sampleMode = isLarge ? "1000" : "all";
      const initTarget = headers[headers.length - 1];
      setCsvModal({
        fileName: file.name, rawRows, headers, naStats,
        selectedTarget: initTarget,
        naStrategy: "drop", sampleMode,
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
  const handleDataConfirm = useCallback((data, features, targetCol, name, totalRows, sampledRows, classLabels, taskType = "classification") => {
    setCustomDataset({ data, features, targetCol, name, totalRows, sampledRows, classLabels, taskType });
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
      style={{ minHeight: "100vh", background: C.bg, color: C.text, fontFamily: "'JetBrains Mono','Fira Code',monospace" }}
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
            fontFamily: "'JetBrains Mono',monospace",
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
          <div style={{ fontSize: 10, color: C.dim, fontFamily: "'JetBrains Mono',monospace" }}>
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
        right={
          <button
            onClick={() => fileInputRef.current?.click()}
            title="Upload a CSV file to use as your dataset"
            onMouseEnter={e => { e.currentTarget.style.color = C.accent; }}
            onMouseLeave={e => { e.currentTarget.style.color = C.dim; }}
            style={{
              background: "none", border: "none", cursor: "pointer",
              fontSize: 11, color: C.dim, fontFamily: "'JetBrains Mono',monospace",
              fontWeight: 600, padding: 0, transition: "color 0.15s",
            }}
          >
            ↑ Upload CSV
          </button>
        }
        detail={
          <div style={{ display: "flex", alignItems: "center", gap: 10, width: "100%" }}>
            <label style={{
              fontSize: 9, color: C.dimmer, fontFamily: "'JetBrains Mono',monospace",
              flexShrink: 0, whiteSpace: "nowrap",
            }}>Dataset</label>
            <select
              value={customDataset ? "custom" : builtinDataset}
              disabled={!!buildProgress || growing}
              onChange={e => {
                const val = e.target.value;
                if (val === "upload") { fileInputRef.current?.click(); return; }
                if (val === "custom") return;
                switchToBuiltin(val);
              }}
              style={{
                ...inp, fontSize: 10, padding: "1px 6px", cursor: (buildProgress || growing) ? "default" : "pointer",
                flex: 1, maxWidth: 340, height: 22,
                opacity: (buildProgress || growing) ? 0.5 : 1,
              }}
            >
              <option value="heart">Heart Disease (binary, {heartData.length} samples)</option>
              <option value="music">Music Genres (10 classes, {musicData.length} samples)</option>
              {customDataset && <option value="custom">Custom: {customDataset.name}</option>}
              <option value="upload">Upload your own CSV…</option>
            </select>
          </div>
        }
      />

      {/* Controls toolbar */}
      <div style={{
        display: "flex", alignItems: "flex-end", gap: 10,
        padding: "12px 20px",
        background: C.panel,
        boxShadow: "0 1px 0 rgba(255,255,255,0.03), 0 4px 20px rgba(0,0,0,0.35)",
        flexWrap: "wrap",
        position: "relative", zIndex: 5,
      }}>
        {/* Model hyperparameters */}
        <div style={{ display: "flex", alignItems: "flex-end", gap: 12, flex: 1, flexWrap: "wrap" }}>
          <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
            <label style={{ fontSize: 9, color: C.dim, fontWeight: 400 }}>Max depth</label>
            <input type="number" min={1} max={activeFeatures.length} value={maxDepthStr} disabled={growing}
              onChange={e => setMaxDepthStr(e.target.value)}
              onBlur={() => {
                const v = Math.max(1, Math.min(activeFeatures.length, parseInt(maxDepthStr, 10) || 1));
                setMaxDepth(v); setMaxDepthStr(String(v));
              }}
              style={{ ...inp, width: 58 }} />
          </div>
          {!lockedMaxFeatures && (
            <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
              <label style={{ fontSize: 9, color: C.dim, fontWeight: 400 }}>Max features</label>
              <select value={featureSubset} disabled={growing} onChange={e => setFeatureSubset(e.target.value)} style={{ ...inp, cursor: "pointer" }}>
                {Object.entries(FEATURE_SUBSET_OPTIONS).map(([k, v]) => (
                  <option key={k} value={k}>{v.label} → {v.fn(activeFeatures.length)}</option>
                ))}
              </select>
            </div>
          )}
          {!lockedNEstimators && (
            <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
              <label style={{ fontSize: 9, color: C.dim, fontWeight: 400 }}>Trees</label>
              <input type="number" min={1} max={100} value={nEstimatorsStr} disabled={growing}
                onChange={e => setNEstimatorsStr(e.target.value)}
                onBlur={() => {
                  const v = Math.max(1, Math.min(100, parseInt(nEstimatorsStr, 10) || 1));
                  setNEstimators(v); setNEstimatorsStr(String(v));
                }}
                style={{ ...inp, width: 58 }} />
            </div>
          )}
          <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
            <label style={{ fontSize: 9, color: C.dim, fontWeight: 400 }}>Speed</label>
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
          <button
            disabled={!!buildProgress}
            onClick={() => {
              if (buildProgress) return;
              if (growing) { cancelRef.current = true; growRef.current = false; setGrowing(false); }
              else autoGrowAll();
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
        {nEstimators > 1 && (
          <div style={{
            flex: 1, display: "flex", alignItems: "center", gap: 3,
            overflowX: "auto", minWidth: 0, padding: "6px 0",
            /* hide scrollbar visually but keep it functional */
            scrollbarWidth: "none", msOverflowStyle: "none",
          }}>
            {trees.map((t, i) => {
              const s       = treeStates[i];
              const done    = t && s && s.stepIdx >= getSteps(i).length - 1 && getSteps(i).length > 0;
              const loading = !t && buildProgress !== null;
              return (
                <button key={i}
                  onClick={() => { if (!growing && !buildProgress && t) setCurTree(i); }}
                  onDoubleClick={() => { if (!growing && !buildProgress && t) { setCurTree(i); instantComplete(i); } }}
                  title={loading ? "Building…" : "Click to view · Double-click to instantly complete"}
                  style={{
                    padding: "3px 10px", borderRadius: 8, border: "none", flexShrink: 0,
                    background: i === curTree ? C.accent : done ? `${C.accent}22` : "rgba(255,255,255,0.04)",
                    color: i === curTree ? "#000" : done ? C.accent : loading ? C.dimmer : C.dim,
                    fontSize: 10, fontFamily: "inherit", cursor: (loading || !t) ? "default" : "pointer",
                    fontWeight: i === curTree ? 700 : 400,
                    transition: "background 0.15s ease-out, color 0.15s ease-out",
                    opacity: loading ? 0.5 : 1,
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
          <button disabled={growing || !!buildProgress || ts.stepIdx <= -1} onClick={() => goToStep(curTree, ts.stepIdx - 1)}
            style={{ ...inp, cursor: (growing || buildProgress || ts.stepIdx <= -1) ? "default" : "pointer", padding: "3px 10px", opacity: (growing || buildProgress || ts.stepIdx <= -1) ? 0.3 : 1, fontSize: 11 }}>◀</button>
          <span style={{ fontSize: 10, color: C.dim, minWidth: 72, textAlign: "center", userSelect: "none" }}>
            {ts.stepIdx === -1 ? "ready" : `${ts.stepIdx + 1} / ${totalSteps}`}
          </span>
          <button disabled={growing || !!buildProgress || ts.stepIdx >= totalSteps - 1} onClick={() => goToStep(curTree, ts.stepIdx + 1)}
            style={{ ...inp, cursor: (growing || buildProgress || ts.stepIdx >= totalSteps - 1) ? "default" : "pointer", padding: "3px 10px", opacity: (growing || buildProgress || ts.stepIdx >= totalSteps - 1) ? 0.3 : 1, fontSize: 11 }}>▶</button>

          {/* Bootstrap info */}
          {curBootstrap && (
            <span style={{ fontSize: 9, color: C.dim, whiteSpace: "nowrap" }}>
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
        <span style={{
          fontSize: 11, fontWeight: 700, color: C.accent,
          letterSpacing: "0.06em", flexShrink: 0,
        }}>
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
                <Edge p1={positions[node.id]} p2={positions[node.left?.id]}  visible={show} label="≤"
                  onPath={edgeLOnPath} sampleActive={samplePath.size > 0} />
                <Edge p1={positions[node.id]} p2={positions[node.right?.id]} visible={show} label=">"
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
          <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 10, flexWrap: "wrap" }}>
            <span style={{ fontSize: 9, color: C.dim, fontWeight: 400 }}>Feature pool</span>
            <span style={{ fontSize: 8.5, color: C.dimmer }}>
              <span style={{ color: C.dimmer }}>● not sampled</span>{"  "}
              <span style={{ color: `${C.accent}99` }}>● candidate</span>{"  "}
              <span style={{ color: C.green }}>● chosen</span>
            </span>
          </div>
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
            } else if (ts.phase >= 1 && isCand) {
              bg = `${C.accent}12`; col = `${C.accent}bb`;
            } else if (ts.phase >= 2 && isGlobalBest) {
              bg = "rgba(255,255,255,0.03)"; col = C.dim;
            } else {
              bg = "rgba(255,255,255,0.03)"; col = C.dimmer;
            }
            return (
              <div key={i} style={{
                padding: "5px 11px", borderRadius: 10, fontSize: 10, fontFamily: "inherit",
                fontWeight: (isBest && ts.phase >= 2) ? 600 : 400,
                background: bg, color: col, boxShadow: shd,
                transition: "all .3s ease-out", minWidth: 80, textAlign: "center",
              }}>
                {f}
                {showGini && (isCand || (isGlobalBest && ts.phase >= 2)) && (
                  <div style={{ fontSize: 7.5, marginTop: 1, opacity: 0.75 }}>
                    {activeTaskType === "regression" ? "MSE" : "G"}={ev.gini.toFixed(3)}
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
        <div style={{ fontSize: 9, color: C.dim, fontWeight: 400, marginBottom: 6 }}>Calculations</div>
        <div style={{
          background: "#0c1018", borderRadius: 12,
          boxShadow: "0 2px 16px rgba(0,0,0,0.35), inset 0 0 0 1px rgba(255,255,255,0.04)",
          overflow: "hidden",
        }}>

          {/* Empty state */}
          {!currentNode && (
            <div style={{ padding: "18px 16px", color: C.dimmer, fontSize: 10.5 }}>
              Press <span style={{ color: C.dim }}>▶ Grow</span> or use <span style={{ color: C.dim }}>→</span> to begin…
            </div>
          )}

          {/* ── Leaf node ──────────────────────────────────────────────────────── */}
          {currentNode?.type === "leaf" && (() => {

            // ── Regression leaf ─────────────────────────────────────────────────
            if (currentNode.mean !== undefined) {
              return (
                <div style={{ padding: "16px 16px" }}>
                  <div style={{ fontSize: 9, color: C.dim, marginBottom: 10 }}>Leaf node · Prediction</div>
                  <div style={{ display: "flex", gap: 20, flexWrap: "wrap", marginBottom: 12 }}>
                    <div>
                      <div style={{ fontSize: 9, color: C.dim, marginBottom: 2 }}>Mean</div>
                      <div style={{ fontSize: 15, fontWeight: 700, color: C.blue, fontFamily: "'JetBrains Mono',monospace" }}>
                        {formatRegVal(currentNode.mean)}
                      </div>
                    </div>
                    <div>
                      <div style={{ fontSize: 9, color: C.dim, marginBottom: 2 }}>Variance</div>
                      <div style={{ fontSize: 12, color: C.text, fontFamily: "'JetBrains Mono',monospace" }}>
                        {currentNode.variance.toFixed(3)}
                      </div>
                    </div>
                    <div>
                      <div style={{ fontSize: 9, color: C.dim, marginBottom: 2 }}>n</div>
                      <div style={{ fontSize: 12, color: C.text, fontFamily: "'JetBrains Mono',monospace" }}>
                        {currentNode.samples}
                      </div>
                    </div>
                  </div>
                  <div style={{ fontSize: 9, color: C.dimmer }}>
                    Range: <span style={{ color: C.dim }}>{formatRegVal(currentNode.min)} – {formatRegVal(currentNode.max)}</span>
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
                  <span style={{ fontSize: 9, color: C.dim }}>Leaf node · Prediction</span>
                </div>
                <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 14 }}>
                  <div style={{
                    fontSize: 15, fontWeight: 700, color: predColor,
                    fontFamily: "'JetBrains Mono',monospace",
                  }}>{currentNode.prediction}</div>
                  <div style={{ fontSize: 9, color: C.dim }}>
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
                  <div style={{ fontSize: 9, color: C.dim, marginBottom: 8 }}>{activeTaskType === "regression" ? "MSE" : "Gini"} per candidate <span style={{ color: C.dimmer }}>(shorter = better)</span></div>
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
                    <span style={{ fontSize: 9, color: C.dimmer, marginLeft: 4 }}>
                      {activeTaskType === "regression" ? "MSE" : "G"}={currentNode.gini.toFixed(4)}
                    </span>
                  </div>
                </div>
              )}
            </>);
          })()}
        </div>
      </div>

      {/* ── Prediction section break — always visible ────────────────────────── */}
      {/* Gradient separator */}
      <div style={{ padding: "20px 20px 0" }}>
        <div style={{
          height: 1,
          background: `linear-gradient(to right, transparent, ${C.accent}28 25%, ${PATH_COLOR}22 75%, transparent)`,
        }} />
      </div>

      {/* Sticky Prediction header */}
      <div style={{
        position: "sticky",
        top: 0,
        zIndex: 8,
        background: C.bg,
        padding: "10px 20px 6px",
        boxShadow: "0 4px 20px rgba(0,0,0,0.5)",
        display: "flex",
        flexDirection: "column",
        gap: 3,
      }}>
        <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
          <span style={{
            fontSize: 11, fontWeight: 700, color: PATH_COLOR,
            letterSpacing: "0.06em", flexShrink: 0,
          }}>
            Prediction
          </span>
          <div style={{ flex: 1, height: 1, background: `linear-gradient(to right, ${PATH_COLOR}40, transparent)` }} />
          <span style={{ fontSize: 9, color: C.dimmer, flexShrink: 0 }}>
            {completedTrees.length}/{nEstimators} trees ready
          </span>
        </div>
        <span style={{ fontSize: 10, color: C.dimmer }}>
          {activeTaskType === "regression"
            ? "Select a sample and see the ensemble's numeric prediction."
            : "Select a sample and watch each tree in the forest vote on it."}
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

            {/* ── Sample selector ─────────────────────────────────────────────── */}
            <div style={{ display: "flex", gap: 8, alignItems: "center", marginBottom: 12 }}>
              <select
                value={safeSampleIdx ?? ""}
                onChange={e => setSelectedSampleIdx(e.target.value === "" ? null : +e.target.value)}
                style={{
                  flex: 1, minWidth: 0,
                  background: "#161c2a", border: "none", color: safeSampleIdx !== null ? C.text : C.dim,
                  fontSize: 10, fontFamily: "'JetBrains Mono',monospace",
                  borderRadius: 8, padding: "5px 10px", outline: "none",
                  boxShadow: "inset 0 0 0 1px rgba(255,255,255,0.06)", cursor: "pointer",
                }}
              >
                <option value="">Select a sample to predict…</option>
                {activeData.map((row, i) => (
                  <option key={i} value={i}>{formatSampleLabel(row, activeFeatures, i)}</option>
                ))}
              </select>
              <button
                onClick={() => setSelectedSampleIdx(Math.floor(Math.random() * activeData.length))}
                style={{
                  flexShrink: 0, padding: "5px 12px", borderRadius: 8, border: "none",
                  background: "rgba(255,255,255,0.05)", color: C.dim, fontSize: 10,
                  fontFamily: "'JetBrains Mono',monospace", cursor: "pointer",
                  boxShadow: "inset 0 0 0 1px rgba(255,255,255,0.06)",
                  transition: "color .15s, background .15s",
                }}
                onMouseEnter={e => { e.currentTarget.style.color = C.text; e.currentTarget.style.background = "rgba(255,255,255,0.09)"; }}
                onMouseLeave={e => { e.currentTarget.style.color = C.dim; e.currentTarget.style.background = "rgba(255,255,255,0.05)"; }}
              >
                Random
              </button>
            </div>

            {/* ── No sample selected ──────────────────────────────────────────── */}
            {!selectedSample && (
              <div style={{ fontSize: 10, color: C.dimmer, paddingBottom: 2 }}>
                Select a sample above to see how each tree predicts it and follow its path through the forest.
              </div>
            )}

            {/* ── Per-tree vote cards ─────────────────────────────────────────── */}
            {selectedSample && (
              <>
                <div style={{ display: "flex", gap: 5, flexWrap: "wrap", alignItems: "center", marginBottom: 12 }}>
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

                  {/* Regression: ensemble mean card */}
                  {activeTaskType === "regression" && sampleMean !== null && (
                    <>
                      <div style={{ fontSize: 10, color: C.dim, marginLeft: 4 }}>→</div>
                      <div style={{ padding: "6px 14px", borderRadius: 10, background: `${C.blue}18`, boxShadow: `inset 0 0 0 1.5px ${C.blue}55` }}>
                        <div style={{ fontSize: 7, color: C.dim, fontWeight: 400, marginBottom: 1 }}>Ensemble mean</div>
                        <div style={{ fontSize: 13, fontWeight: 800, color: C.blue, fontFamily: "'JetBrains Mono',monospace" }}>
                          {formatRegVal(sampleMean)}
                        </div>
                      </div>
                    </>
                  )}

                  {/* Classification: majority vote card */}
                  {activeTaskType === "classification" && sampleMajority && (
                    <>
                      <div style={{ fontSize: 10, color: C.dim, marginLeft: 4 }}>→</div>
                      <div style={{
                        padding: "6px 14px", borderRadius: 10,
                        background: `${classColor(sampleMajority, classLabels)}18`,
                        boxShadow: `inset 0 0 0 1.5px ${classColor(sampleMajority, classLabels)}55`,
                      }}>
                        <div style={{ fontSize: 7, color: C.dim, fontWeight: 400, marginBottom: 1 }}>Majority vote</div>
                        <div style={{ fontSize: 13, fontWeight: 800, color: classColor(sampleMajority, classLabels) }}>
                          {sampleMajority}
                        </div>
                      </div>
                    </>
                  )}
                </div>

                {/* ── Regression result breakdown ──────────────────────────────── */}
                {activeTaskType === "regression" && sampleMean !== null && (
                  <div style={{ marginBottom: 10, fontSize: 10, color: C.dim, lineHeight: 1.8 }}>
                    <span>Ensemble mean: <strong style={{ color: C.blue, fontFamily: "'JetBrains Mono',monospace" }}>{formatRegVal(sampleMean)}</strong></span>
                    <span style={{ color: C.dimmer }}> ({completedTrees.length} tree{completedTrees.length !== 1 ? "s" : ""})</span>
                    {sampleTrueLabel !== null && (<>
                      <span style={{ color: C.dimmer }}> · True: </span>
                      <strong style={{ fontFamily: "'JetBrains Mono',monospace", color: C.text }}>{formatRegVal(sampleTrueLabel)}</strong>
                      <span style={{ color: C.dimmer }}> · Error: </span>
                      <strong style={{ fontFamily: "'JetBrains Mono',monospace", color: Math.abs(sampleMean - sampleTrueLabel) / (Math.abs(sampleTrueLabel) || 1) < 0.1 ? C.green : C.orange }}>
                        {formatRegVal(Math.abs(sampleMean - sampleTrueLabel))}
                      </strong>
                    </>)}
                  </div>
                )}

                {/* ── Classification vote breakdown + correctness ───────────────── */}
                {activeTaskType === "classification" && sampleMajority && (
                  <div style={{ marginBottom: 10 }}>
                    <div style={{ fontSize: 10, color: C.dim, marginBottom: 5, lineHeight: 1.6 }}>
                      {classLabels.map((cls, i, arr) => {
                        const n = sampleVotesPerClass[cls] ?? 0;
                        if (n === 0) return null;
                        return (
                          <span key={cls}>
                            <strong style={{ color: classColor(cls, classLabels) }}>{n}</strong>
                            <span style={{ color: C.dimmer }}>/{completedTrees.length} </span>
                            <span style={{ color: classColor(cls, classLabels) }}>{cls}</span>
                            {i < arr.length - 1 && <span style={{ color: C.dimmer }}> · </span>}
                          </span>
                        );
                      })}
                      <span style={{ marginLeft: 8, color: C.dimmer }}>True: </span>
                      <strong style={{ color: classColor(sampleTrueLabel, classLabels) }}>{sampleTrueLabel}</strong>
                      {" "}
                      <span style={{ color: sampleCorrect ? C.green : C.red, fontWeight: 700 }}>
                        {sampleCorrect ? "✓" : "✗"}
                      </span>
                    </div>
                    <div style={{ height: 6, borderRadius: 3, overflow: "hidden", background: "#151a24", display: "flex" }}>
                      {classLabels.map(cls => {
                        const pct = ((sampleVotesPerClass[cls] ?? 0) / completedTrees.length) * 100;
                        return pct > 0
                          ? <div key={cls} style={{ width: `${pct}%`, background: classColor(cls, classLabels), transition: "width .3s" }} />
                          : null;
                      })}
                    </div>
                  </div>
                )}
              </>
            )}

            {/* ── Accuracy summary ────────────────────────────────────────────── */}
            {/* Classification */}
            {activeTaskType === "classification" && (forestAccuracy !== null || avgOobAccuracy !== null) && (
              <div style={{ marginTop: selectedSample ? 4 : 0, paddingTop: 10, borderTop: "1px solid rgba(255,255,255,0.05)", fontSize: 9.5, color: C.dimmer, display: "flex", gap: 16, flexWrap: "wrap" }}>
                {forestAccuracy !== null && (
                  <span>Forest accuracy: <strong style={{ color: C.dim }}>{forestAccuracy}%</strong><span style={{ color: C.dimmer }}> (training)</span></span>
                )}
                {avgOobAccuracy !== null && (
                  <span style={{ display: "flex", alignItems: "center", gap: 4, position: "relative" }}>
                    OOB accuracy: <strong style={{ color: C.dim }}>{avgOobAccuracy}%</strong>
                    <span style={{ color: C.dimmer }}> (unseen)</span>
                    <span style={{ cursor: "help", color: C.dimmer, userSelect: "none" }}
                      onMouseEnter={() => setOobTooltipVisible(true)}
                      onMouseLeave={() => setOobTooltipVisible(false)}> ⓘ</span>
                    {oobTooltipVisible && (
                      <div style={{ position: "absolute", bottom: "calc(100% + 6px)", left: 0, width: 240, background: "#141b2d", borderRadius: 9, padding: "9px 12px", fontSize: 9.5, color: C.dim, lineHeight: 1.6, boxShadow: "0 8px 32px rgba(0,0,0,0.6), inset 0 0 0 1px rgba(255,255,255,0.08)", zIndex: 20, pointerEvents: "none" }}>
                        Each tree is tested on the ~37% of samples not used in its bootstrap training set.
                        This gives an unbiased estimate of generalization error with no held-out test set needed.
                      </div>
                    )}
                  </span>
                )}
              </div>
            )}
            {/* Regression */}
            {activeTaskType === "regression" && (forestRegMetrics !== null || avgOobR2 !== null) && (
              <div style={{ marginTop: selectedSample ? 4 : 0, paddingTop: 10, borderTop: "1px solid rgba(255,255,255,0.05)", fontSize: 9.5, color: C.dimmer, display: "flex", gap: 16, flexWrap: "wrap" }}>
                {forestRegMetrics !== null && (
                  <span>
                    R²: <strong style={{ color: C.dim }}>{forestRegMetrics.r2}</strong>
                    {" · "}RMSE: <strong style={{ color: C.dim }}>{forestRegMetrics.rmse}</strong>
                    <span style={{ color: C.dimmer }}> (training)</span>
                  </span>
                )}
                {avgOobR2 !== null && (
                  <span style={{ display: "flex", alignItems: "center", gap: 4, position: "relative" }}>
                    OOB R²: <strong style={{ color: C.dim }}>{avgOobR2}</strong>
                    {" · "}RMSE: <strong style={{ color: C.dim }}>{avgOobRMSE}</strong>
                    <span style={{ color: C.dimmer }}> (unseen)</span>
                    <span style={{ cursor: "help", color: C.dimmer, userSelect: "none" }}
                      onMouseEnter={() => setOobTooltipVisible(true)}
                      onMouseLeave={() => setOobTooltipVisible(false)}> ⓘ</span>
                    {oobTooltipVisible && (
                      <div style={{ position: "absolute", bottom: "calc(100% + 6px)", left: 0, width: 260, background: "#141b2d", borderRadius: 9, padding: "9px 12px", fontSize: 9.5, color: C.dim, lineHeight: 1.6, boxShadow: "0 8px 32px rgba(0,0,0,0.6), inset 0 0 0 1px rgba(255,255,255,0.08)", zIndex: 20, pointerEvents: "none" }}>
                        R² measures proportion of variance explained. 1.0 is perfect, 0.0 is no better than predicting the mean.
                        OOB estimates this on the ~37% of samples held out from each tree's bootstrap.
                      </div>
                    )}
                  </span>
                )}
              </div>
            )}

          </>)}
        </div>
      </div>

      {/* Footer */}
      <div style={{ padding: "8px 16px", boxShadow: "0 -1px 0 rgba(255,255,255,0.04)", fontSize: 8.5, color: C.dimmer, textAlign: "center" }}>
        max_features={subsetSize}/{activeFeatures.length} · Trees: {nEstimators} · Bootstrap n={TOTAL_SAMPLES}
      </div>
    </div>
  );
}
