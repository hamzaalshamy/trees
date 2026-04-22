/* eslint-disable react-refresh/only-export-components */
import { C, MONO } from "./theme";

// ─── Class color palette ───────────────────────────────────────────────────────
// Index 0 = blue, index 1 = pink — preserves binary heart-disease appearance.
// For 3+ classes the remaining slots provide visually distinct hues.
export const CLASS_PALETTE = [
  C.blue, C.leafB, C.green, C.accent, C.purple,
  "#06b6d4", "#f472b6", "#84cc16", "#fb923c", "#a3e635",
];

export function classColor(cls, allClasses) {
  const idx = allClasses.indexOf(cls);
  return idx >= 0 ? CLASS_PALETTE[idx % CLASS_PALETTE.length] : C.dim;
}

// ─── Tree node helpers ─────────────────────────────────────────────────────────

export function flattenNodes(node, id = "0") {
  node.id = id;
  const nodes = [node];
  if (node.type === "split") {
    nodes.push(...flattenNodes(node.left,  id + "L"));
    nodes.push(...flattenNodes(node.right, id + "R"));
  }
  return nodes;
}

// Aggregate across all leaves: majority class (classification) or weighted mean (regression).
export function getTreePrediction(node) {
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

// Format a split threshold for display on edges.
export function fmtThresh(v) {
  if (v == null) return "?";
  const abs = Math.abs(v);
  if (abs >= 1000) return v.toFixed(0);
  if (abs >= 10)   return v.toFixed(1);
  if (abs >= 1)    return v.toFixed(2);
  return v.toFixed(3);
}

// Count actual leaf nodes so width scales with real tree shape, not worst-case depth.
export function countLeaves(node) {
  if (!node || node.type === "leaf") return 1;
  return countLeaves(node.left) + countLeaves(node.right);
}

export const LEAF_SPACING = 130; // px between leaf centres
export const X_PAD = 30;         // left/right margin
export const Y_GAP = 90;         // vertical distance between depth levels

export function computeTreeWidth(node) {
  return X_PAD * 2 + countLeaves(node) * LEAF_SPACING;
}

// Positions each node so horizontal space is proportional to its subtree's leaf count.
export function computePositions(node) {
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
export function splitFeatureName(name, maxChars = 22) {
  if (name.length <= maxChars) return { line1: name, line2: null };
  const mid = Math.floor(name.length / 2);
  let breakAt = -1;
  for (let d = 0; d <= mid; d++) {
    if (name[mid - d] === " ") { breakAt = mid - d; break; }
    if (name[mid + d] === " ") { breakAt = mid + d; break; }
  }
  if (breakAt === -1) {
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
export function formatRegVal(v) {
  if (v === null || v === undefined || (typeof v === "number" && isNaN(v))) return "?";
  const abs = Math.abs(v);
  if (abs >= 1e6)  return v.toLocaleString(undefined, { maximumFractionDigits: 0 });
  if (abs >= 1000) return v.toLocaleString(undefined, { maximumFractionDigits: 1 });
  if (abs >= 10)   return v.toFixed(2);
  return v.toFixed(3);
}

// Returns ordered array of node IDs from root to the leaf a sample lands in.
export function getSamplePath(node, row, features) {
  const path = [node.id];
  if (node.type === "leaf") return path;
  const fname = features[node.featureIndex];
  const goLeft = row[fname] <= node.threshold;
  return [...path, ...getSamplePath(goLeft ? node.left : node.right, row, features)];
}

// Formats a data row as a compact label for the sample dropdown.
export function formatSampleLabel(row, features, idx) {
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
export const PATH_COLOR = "#22d3ee";

export function Edge({ p1, p2, visible, label, onPath, sampleActive }) {
  if (!visible || !p1 || !p2) return null;
  const highlighted = sampleActive && onPath;
  const dimmed      = sampleActive && !onPath;
  const x1 = p1.x, y1 = p1.y + 27;
  const x2 = p2.x, y2 = p2.y - 24;
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
      <rect x={lx - labelW / 2} y={ly - 7} width={labelW} height={13}
        rx={3} fill={C.bg} opacity={0.88} />
      <text x={lx} y={ly + 2}
        textAnchor="middle" dominantBaseline="middle"
        fill={textCol} fontSize={9} fontFamily={MONO}
        fontWeight={highlighted ? 700 : 400}>
        {label}
      </text>
    </g>
  );
}

export function TreeNode({ node, show, phase, pos, allClasses, onPath, sampleActive, isRegression }) {
  if (!show || !pos) return null;
  const { x, y } = pos;
  const highlighted = sampleActive && onPath;
  const dimmed      = sampleActive && !onPath;

  // ── Leaf node ─────────────────────────────────────────────────────────────
  if (node.type === "leaf") {
    const vis = phase >= 1 ? 1 : 0;
    const gStyle = {
      opacity: dimmed ? 0.2 : vis, transition: "opacity .45s ease-out",
      transformOrigin: `${x}px ${y}px`,
      // Don't play nodeIn for dimmed nodes: they should appear at opacity 0.2
      // immediately, not animate from 0→1 then snap back to 0.2.
      animation: (vis && !dimmed) ? "nodeIn 0.35s ease-out" : "none",
    };

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
            fontFamily={MONO} fontWeight={700}>{meanStr}</text>
          <text x={x} y={y + 3} textAnchor="middle" fill={C.dim} fontSize={7.5}
            fontFamily={MONO}>n={node.samples}</text>
          <text x={x} y={y + 16} textAnchor="middle" fill={C.dim} fontSize={7}
            fontFamily={MONO}>{minStr}–{maxStr}</text>
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
    // eslint-disable-next-line react-hooks/immutability
    let xCursor = barX0;
    const barSegs = allClasses.map(cls => {
      const count = node.classCounts?.[cls] ?? 0;
      const w = node.samples > 0 ? Math.max(0, (count / node.samples) * BAR_W) : 0;
      // eslint-disable-next-line react-hooks/immutability
      const rx = xCursor; xCursor += w;
      return w > 0 ? <rect key={cls} x={rx} y={y + 2} width={w} height={5} fill={classColor(cls, allClasses)} /> : null;
    });

    return (
      <g style={gStyle}>
        <rect x={x - 38} y={y - 24} width={76} height={50} rx={11}
          fill={C.panel} stroke={leafStroke} strokeWidth={highlighted ? 2 : 1.4} filter={leafFilter} />
        <text x={x} y={y - 9} textAnchor="middle" fill={highlighted ? PATH_COLOR : C.text} fontSize={9}
          fontFamily={MONO} fontWeight={600}>{displayLabel}</text>
        <rect x={barX0} y={y + 2} width={BAR_W} height={5} rx={2.5} fill="rgba(255,255,255,0.06)" />
        {barSegs}
        <text x={x} y={y + 17} textAnchor="middle" fill={C.dim} fontSize={7.5}
          fontFamily={MONO}>n={node.samples} G={node.impurity.toFixed(3)}</text>
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
          fontFamily={MONO} fontWeight={700}>?</text>
        <text x={x} y={y + 11} textAnchor="middle" fill={C.dim} fontSize={7.5}
          fontFamily={MONO}>n={node.samples}</text>
      </g>
      <g style={{ opacity: revealed ? 1 : 0, transition: "opacity 0.3s" }}>
        {line2 ? (
          <>
            <text x={x} y={y - 13} textAnchor="middle" fill={C.accent} fontSize={8.5}
              fontFamily={MONO} fontWeight={700}>{line1}</text>
            <text x={x} y={y - 2} textAnchor="middle" fill={C.accent} fontSize={8.5}
              fontFamily={MONO} fontWeight={700}>{line2}</text>
            <text x={x} y={y + 14} textAnchor="middle" fill={C.dim} fontSize={7.5}
              fontFamily={MONO}>≤{node.threshold} {crit}={node.gini.toFixed(3)} n={node.samples}</text>
          </>
        ) : (
          <>
            <text x={x} y={y - 6} textAnchor="middle" fill={C.accent} fontSize={9}
              fontFamily={MONO} fontWeight={700}>{line1}</text>
            <text x={x} y={y + 10} textAnchor="middle" fill={C.dim} fontSize={7.5}
              fontFamily={MONO}>≤{node.threshold} {crit}={node.gini.toFixed(3)} n={node.samples}</text>
          </>
        )}
      </g>
    </g>
  );
}
