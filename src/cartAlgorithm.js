// Real CART (Classification And Regression Trees) implementation.
// Fully general: supports any number of classes — binary is just the n=2 case.
// Also supports regression via MSE reduction.

function shuffleArray(arr) {
  const a = [...arr];
  for (let i = a.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [a[i], a[j]] = [a[j], a[i]];
  }
  return a;
}

// ─── Classification helpers ────────────────────────────────────────────────────

// Count occurrences of each class value in data[targetCol].
function countClasses(data, targetCol) {
  const counts = {};
  for (const row of data) {
    const cls = row[targetCol];
    counts[cls] = (counts[cls] ?? 0) + 1;
  }
  return counts;
}

// General Gini: 1 − Σpᵢ². Works for any number of classes.
function giniFromCounts(counts, total) {
  if (total === 0) return 0;
  let sq = 0;
  for (const c of Object.values(counts)) sq += (c / total) ** 2;
  return 1 - sq;
}

// Weighted Gini after a split — same formula, just applied to each child.
function weightedGini(leftCounts, nL, rightCounts, nR) {
  const n = nL + nR;
  if (n === 0) return 0;
  return (nL / n) * giniFromCounts(leftCounts, nL)
       + (nR / n) * giniFromCounts(rightCounts, nR);
}

// ─── Regression helpers ────────────────────────────────────────────────────────

function colMean(data, targetCol) {
  if (!data.length) return 0;
  return data.reduce((s, r) => s + r[targetCol], 0) / data.length;
}

function colVariance(data, targetCol) {
  if (data.length < 2) return 0;
  const m = colMean(data, targetCol);
  return data.reduce((s, r) => s + (r[targetCol] - m) ** 2, 0) / data.length;
}

// Weighted MSE (variance) after a split.
function weightedVariance(leftData, rightData, targetCol) {
  const n = leftData.length + rightData.length;
  if (!n) return 0;
  return (leftData.length / n) * colVariance(leftData, targetCol)
       + (rightData.length / n) * colVariance(rightData, targetCol);
}

// ─── Feature evaluation ────────────────────────────────────────────────────────

// Evaluate one feature: find the threshold that minimises the split criterion.
// Returns { featureIndex, gini, threshold } — "gini" holds MSE for regression.
function evalFeatureOnData(data, featureIndex, features, targetCol, mode) {
  const fname  = features[featureIndex];
  const sorted = [...new Set(data.map(r => r[fname]))].sort((a, b) => a - b);

  if (mode === "regression") {
    if (sorted.length < 2) {
      return { featureIndex, gini: colVariance(data, targetCol), threshold: sorted[0] ?? 0 };
    }
    let bestMSE = Infinity, bestThreshold = sorted[0];
    for (let k = 0; k < sorted.length - 1; k++) {
      const t = (sorted[k] + sorted[k + 1]) / 2;
      const left = [], right = [];
      for (const row of data) {
        if (row[fname] <= t) left.push(row); else right.push(row);
      }
      const mse = weightedVariance(left, right, targetCol);
      if (mse < bestMSE) { bestMSE = mse; bestThreshold = t; }
    }
    return { featureIndex, gini: bestMSE, threshold: +bestThreshold.toFixed(4) };
  }

  // Classification — Gini criterion
  if (sorted.length < 2) {
    const counts = countClasses(data, targetCol);
    return { featureIndex, gini: giniFromCounts(counts, data.length), threshold: sorted[0] ?? 0 };
  }

  let bestGini = Infinity, bestThreshold = sorted[0];
  for (let k = 0; k < sorted.length - 1; k++) {
    const t = (sorted[k] + sorted[k + 1]) / 2;
    const lCounts = {}, rCounts = {};
    let nL = 0, nR = 0;
    for (const row of data) {
      const cls = row[targetCol];
      if (row[fname] <= t) { lCounts[cls] = (lCounts[cls] ?? 0) + 1; nL++; }
      else                  { rCounts[cls] = (rCounts[cls] ?? 0) + 1; nR++; }
    }
    const g = weightedGini(lCounts, nL, rCounts, nR);
    if (g < bestGini) { bestGini = g; bestThreshold = t; }
  }
  return { featureIndex, gini: bestGini, threshold: +bestThreshold.toFixed(4) };
}

// ─── Tree builder ──────────────────────────────────────────────────────────────

// Recursive CART builder. Works for classification and regression.
//
// Classification leaf shape:
//   { type:"leaf", samples, classCounts, impurity, prediction, depth }
//   prediction = class label string with highest count
//
// Regression leaf shape:
//   { type:"leaf", samples, mean, variance, min, max, impurity, prediction, depth }
//   prediction = mean (number), impurity = variance
export function buildRealTree(data, features, targetCol, maxDepth, subsetSize, depth = 0, mode = "classification") {
  const n = data.length;

  if (mode === "regression") {
    const vals   = data.map(r => r[targetCol]);
    const mean   = vals.reduce((s, v) => s + v, 0) / n;
    const vari   = n > 1 ? vals.reduce((s, v) => s + (v - mean) ** 2, 0) / n : 0;
    const minVal = Math.min(...vals);
    const maxVal = Math.max(...vals);

    if (depth >= maxDepth || n <= 5 || vari < 1e-10) {
      return { type: "leaf", samples: n, mean, variance: vari, min: minVal, max: maxVal, impurity: vari, prediction: mean, depth };
    }

    const allIdx          = features.map((_, i) => i);
    const allFeatureEvals = allIdx.map(fi => evalFeatureOnData(data, fi, features, targetCol, "regression"));
    const candidateIndices = shuffleArray(allIdx).slice(0, subsetSize);

    let bestFeature = candidateIndices[0], bestGini = Infinity, bestThreshold = 0;
    for (const fi of candidateIndices) {
      const ev = allFeatureEvals[fi];
      if (ev.gini < bestGini) { bestGini = ev.gini; bestFeature = fi; bestThreshold = ev.threshold; }
    }

    let globalBestIdx = 0, globalBestGini = Infinity;
    for (const ev of allFeatureEvals) {
      if (ev.gini < globalBestGini) { globalBestGini = ev.gini; globalBestIdx = ev.featureIndex; }
    }

    const fname     = features[bestFeature];
    const leftData  = data.filter(r => r[fname] <= bestThreshold);
    const rightData = data.filter(r => r[fname] >  bestThreshold);

    if (!leftData.length || !rightData.length) {
      return { type: "leaf", samples: n, mean, variance: vari, min: minVal, max: maxVal, impurity: vari, prediction: mean, depth };
    }

    return {
      type: "split",
      featureIndex: bestFeature,
      featureName:  features[bestFeature],
      threshold:    bestThreshold,
      gini:         bestGini, // holds MSE for regression
      samples:      n,
      candidateIndices,
      allFeatureEvals,
      globalBestIdx,
      globalBestGini,
      depth,
      left:  buildRealTree(leftData,  features, targetCol, maxDepth, subsetSize, depth + 1, "regression"),
      right: buildRealTree(rightData, features, targetCol, maxDepth, subsetSize, depth + 1, "regression"),
    };
  }

  // ── Classification ──────────────────────────────────────────────────────────
  const classCounts = countClasses(data, targetCol);
  const impurity    = giniFromCounts(classCounts, n);
  const prediction  = Object.entries(classCounts).sort((a, b) => b[1] - a[1])[0]?.[0] ?? "?";

  const nDistinct = Object.values(classCounts).filter(c => c > 0).length;
  if (depth >= maxDepth || n <= 5 || nDistinct <= 1) {
    return { type: "leaf", samples: n, classCounts, impurity, prediction, depth };
  }

  const allIdx          = features.map((_, i) => i);
  const allFeatureEvals = allIdx.map(fi => evalFeatureOnData(data, fi, features, targetCol, "classification"));
  const candidateIndices = shuffleArray(allIdx).slice(0, subsetSize);

  let bestFeature = candidateIndices[0], bestGini = Infinity, bestThreshold = 0;
  for (const fi of candidateIndices) {
    const ev = allFeatureEvals[fi];
    if (ev.gini < bestGini) { bestGini = ev.gini; bestFeature = fi; bestThreshold = ev.threshold; }
  }

  let globalBestIdx = 0, globalBestGini = Infinity;
  for (const ev of allFeatureEvals) {
    if (ev.gini < globalBestGini) { globalBestGini = ev.gini; globalBestIdx = ev.featureIndex; }
  }

  const fname     = features[bestFeature];
  const leftData  = data.filter(r => r[fname] <= bestThreshold);
  const rightData = data.filter(r => r[fname] >  bestThreshold);

  if (leftData.length === 0 || rightData.length === 0) {
    return { type: "leaf", samples: n, classCounts, impurity, prediction, depth };
  }

  return {
    type: "split",
    featureIndex: bestFeature,
    featureName:  features[bestFeature],
    threshold:    bestThreshold,
    gini:         bestGini,
    samples:      n,
    candidateIndices,
    allFeatureEvals,
    globalBestIdx,
    globalBestGini,
    depth,
    left:  buildRealTree(leftData,  features, targetCol, maxDepth, subsetSize, depth + 1, "classification"),
    right: buildRealTree(rightData, features, targetCol, maxDepth, subsetSize, depth + 1, "classification"),
  };
}

// ─── Bootstrap sampling ────────────────────────────────────────────────────────

// Sample n rows with replacement; return bootstrapData + oobIndices.
export function bootstrapSample(data) {
  const n = data.length;
  const drawnSet = new Set();
  const bootstrapData = [];
  for (let i = 0; i < n; i++) {
    const idx = Math.floor(Math.random() * n);
    drawnSet.add(idx);
    bootstrapData.push(data[idx]);
  }
  const oobIndices = [];
  for (let i = 0; i < n; i++) { if (!drawnSet.has(i)) oobIndices.push(i); }
  return { bootstrapData, oobIndices, inBag: drawnSet.size };
}

// ─── Prediction ────────────────────────────────────────────────────────────────

// Traverse tree for a single row.
// Classification: returns the predicted class label string.
// Regression:     returns the leaf mean (number).
export function predictRow(tree, row, features) {
  if (tree.type === "leaf") return tree.prediction;
  const fname = features[tree.featureIndex];
  return row[fname] <= tree.threshold
    ? predictRow(tree.left,  row, features)
    : predictRow(tree.right, row, features);
}

// ─── OOB metrics ───────────────────────────────────────────────────────────────

// Classification OOB accuracy (proportion correct).
export function computeOOBAccuracy(tree, data, oobIndices, features, targetCol) {
  if (oobIndices.length === 0) return null;
  let correct = 0;
  for (const idx of oobIndices) {
    const pred   = predictRow(tree, data[idx], features);
    const actual = data[idx][targetCol];
    if (pred === actual) correct++;
  }
  return +(correct / oobIndices.length).toFixed(3);
}

// Regression OOB metrics: R² and RMSE.
export function computeOOBMetricsRegression(tree, data, oobIndices, features, targetCol) {
  if (oobIndices.length === 0) return null;
  const actual = oobIndices.map(i => data[i][targetCol]);
  const pred   = oobIndices.map(i => predictRow(tree, data[i], features));
  const n      = actual.length;
  const meanA  = actual.reduce((s, v) => s + v, 0) / n;
  const ssTot  = actual.reduce((s, v) => s + (v - meanA) ** 2, 0);
  const ssRes  = actual.reduce((s, v, i) => s + (v - pred[i]) ** 2, 0);
  const r2     = ssTot > 0 ? 1 - ssRes / ssTot : 1;
  const rmse   = Math.sqrt(ssRes / n);
  return { r2: +r2.toFixed(4), rmse: +rmse.toFixed(4) };
}
