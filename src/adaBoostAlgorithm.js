// AdaBoost (SAMME / AdaBoost.M1) with weighted CART trees.
// Trees are built on the original data but using per-sample weights
// to adjust the Gini impurity criterion.

// Assign IDs to every node (same format as cartAlgorithm for rendering compat).
export function flattenNodes(node, id = "0") {
  node.id = id;
  const nodes = [node];
  if (node.type === "split") {
    nodes.push(...flattenNodes(node.left,  id + "L"));
    nodes.push(...flattenNodes(node.right, id + "R"));
  }
  return nodes;
}

// ─── Weighted Gini helpers ──────────────────────────────────────────────────────

function wGini(indices, weights, targets) {
  let totalW = 0;
  const cw = {};
  for (const i of indices) {
    totalW += weights[i];
    cw[targets[i]] = (cw[targets[i]] ?? 0) + weights[i];
  }
  if (totalW === 0) return 0;
  let sq = 0;
  for (const w of Object.values(cw)) sq += (w / totalW) ** 2;
  return 1 - sq;
}

// Evaluate one feature: best threshold by weighted Gini.
function evalWeighted(indices, fname, data, weights, targets) {
  const vals = [...new Set(indices.map(i => data[i][fname]))].sort((a, b) => a - b);
  if (vals.length < 2) {
    return { gini: wGini(indices, weights, targets), threshold: vals[0] ?? 0 };
  }
  let bestG = Infinity, bestT = vals[0];
  for (let k = 0; k < vals.length - 1; k++) {
    const t = (vals[k] + vals[k + 1]) / 2;
    let wL = 0, wR = 0;
    const cwL = {}, cwR = {};
    for (const i of indices) {
      const w = weights[i], cls = targets[i];
      if (data[i][fname] <= t) { wL += w; cwL[cls] = (cwL[cls] ?? 0) + w; }
      else                      { wR += w; cwR[cls] = (cwR[cls] ?? 0) + w; }
    }
    if (wL === 0 || wR === 0) continue;
    const wT = wL + wR;
    let gL = 1, gR = 1;
    for (const w of Object.values(cwL)) gL -= (w / wL) ** 2;
    for (const w of Object.values(cwR)) gR -= (w / wR) ** 2;
    const g = (wL / wT) * gL + (wR / wT) * gR;
    if (g < bestG) { bestG = g; bestT = t; }
  }
  return { gini: +bestG.toFixed(4), threshold: +bestT.toFixed(4) };
}

function sampleCounts(indices, targets) {
  const c = {};
  for (const i of indices) c[targets[i]] = (c[targets[i]] ?? 0) + 1;
  return c;
}

function majorityByWeight(indices, weights, targets) {
  const cw = {};
  for (const i of indices) cw[targets[i]] = (cw[targets[i]] ?? 0) + weights[i];
  return Object.entries(cw).sort((a, b) => b[1] - a[1])[0]?.[0] ?? "?";
}

// ─── Weighted CART ──────────────────────────────────────────────────────────────

// Returns the same node shape as cartAlgorithm.buildRealTree so existing
// Edge / TreeNode SVG components render it without modification.
function buildWeightedTree(indices, data, weights, targets, features, maxDepth, depth = 0) {
  const n = indices.length;
  const classCounts = sampleCounts(indices, targets);
  const impurity    = wGini(indices, weights, targets);
  const prediction  = majorityByWeight(indices, weights, targets);
  const nDistinct   = Object.values(classCounts).filter(c => c > 0).length;

  if (depth >= maxDepth || n <= 1 || nDistinct <= 1) {
    return { type: "leaf", samples: n, classCounts, impurity, prediction, depth };
  }

  // AdaBoost uses ALL features at every split (no subsampling).
  const allFeatureEvals = features.map((fname, fi) => {
    const ev = evalWeighted(indices, fname, data, weights, targets);
    return { featureIndex: fi, gini: ev.gini, threshold: ev.threshold };
  });
  const candidateIndices = features.map((_, i) => i);

  let bestFi = 0, bestG = Infinity, bestT = 0;
  for (const ev of allFeatureEvals) {
    if (ev.gini < bestG) { bestG = ev.gini; bestFi = ev.featureIndex; bestT = ev.threshold; }
  }

  const fname    = features[bestFi];
  const leftIdx  = indices.filter(i => data[i][fname] <= bestT);
  const rightIdx = indices.filter(i => data[i][fname] >  bestT);

  if (!leftIdx.length || !rightIdx.length) {
    return { type: "leaf", samples: n, classCounts, impurity, prediction, depth };
  }

  return {
    type: "split",
    featureIndex: bestFi, featureName: features[bestFi],
    threshold: bestT, gini: bestG, samples: n,
    candidateIndices, allFeatureEvals,
    globalBestIdx: bestFi, globalBestGini: bestG,
    classCounts, depth,
    left:  buildWeightedTree(leftIdx,  data, weights, targets, features, maxDepth, depth + 1),
    right: buildWeightedTree(rightIdx, data, weights, targets, features, maxDepth, depth + 1),
  };
}

// ─── Prediction ─────────────────────────────────────────────────────────────────

export function predictAdaTree(tree, row, features) {
  if (tree.type === "leaf") return tree.prediction;
  const fname = features[tree.featureIndex];
  return row[fname] <= tree.threshold
    ? predictAdaTree(tree.left,  row, features)
    : predictAdaTree(tree.right, row, features);
}

// Returns { class: alphaSum } for ensemble prediction.
export function adaBoostEnsembleScores(rounds, row, features) {
  const scores = {};
  for (const { tree, alpha } of rounds) {
    const pred = predictAdaTree(tree, row, features);
    scores[pred] = (scores[pred] ?? 0) + alpha;
  }
  return scores;
}

// Training accuracy of the ensemble on first k rounds.
export function computeEnsembleAccuracy(rounds, data, features, targetCol) {
  if (!rounds.length || !data.length) return null;
  let correct = 0;
  for (const row of data) {
    const scores = adaBoostEnsembleScores(rounds, row, features);
    const pred = Object.entries(scores).sort((a, b) => b[1] - a[1])[0]?.[0];
    if (pred === row[targetCol]) correct++;
  }
  return +(correct / data.length * 100).toFixed(1);
}

// ─── Full AdaBoost run ──────────────────────────────────────────────────────────

export function runAdaBoost(data, features, targetCol, nRounds, maxDepth) {
  const n       = data.length;
  const targets = data.map(r => r[targetCol]);
  const weights = new Array(n).fill(1 / n);
  const rounds  = [];

  for (let r = 0; r < nRounds; r++) {
    const indices = data.map((_, i) => i);
    const tree    = buildWeightedTree(indices, data, [...weights], targets, features, maxDepth);
    flattenNodes(tree);

    const predictions   = data.map(row => predictAdaTree(tree, row, features));
    const misclassified = predictions.map((pred, i) => pred !== targets[i]);

    let weightedError = misclassified.reduce((s, m, i) => s + (m ? weights[i] : 0), 0);
    weightedError = Math.max(1e-10, Math.min(1 - 1e-10, weightedError));

    const alpha = 0.5 * Math.log((1 - weightedError) / weightedError);

    const newWeights = weights.map((w, i) => w * Math.exp(misclassified[i] ? alpha : -alpha));
    const wSum       = newWeights.reduce((s, w) => s + w, 0);
    const normalized = newWeights.map(w => w / wSum);

    rounds.push({
      tree,
      alpha:        +alpha.toFixed(4),
      error:        +weightedError.toFixed(4),
      weightsInput: [...weights],
      weightsOutput: normalized,
      predictions,
      misclassified,
    });

    for (let i = 0; i < n; i++) weights[i] = normalized[i];
  }

  // Precompute per-round cumulative ensemble accuracy.
  const cumulativeAccuracy = rounds.map((_, k) =>
    computeEnsembleAccuracy(rounds.slice(0, k + 1), data, features, targetCol)
  );

  return { rounds, cumulativeAccuracy };
}
