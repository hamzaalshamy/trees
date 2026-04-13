// Real CART (Classification And Regression Trees) implementation.
// Produces trees whose node shape exactly matches what RandomForestViz.jsx expects.

function shuffleArray(arr) {
  const a = [...arr];
  for (let i = a.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [a[i], a[j]] = [a[j], a[i]];
  }
  return a;
}

function weightedGini(lA, lB, rA, rB) {
  const nL = lA + lB, nR = rA + rB, n = nL + nR;
  if (n === 0) return 0;
  const gL = nL > 0 ? 1 - (lA / nL) ** 2 - (lB / nL) ** 2 : 0;
  const gR = nR > 0 ? 1 - (rA / nR) ** 2 - (rB / nR) ** 2 : 0;
  return (nL / n) * gL + (nR / n) * gR;
}

// Evaluate one feature: find the best threshold by scanning all midpoints between
// consecutive unique sorted values. Returns { featureIndex, gini, threshold }.
function evalFeatureOnData(data, featureIndex, features, targetCol) {
  const fname = features[featureIndex];
  const sorted = [...new Set(data.map(r => r[fname]))].sort((a, b) => a - b);

  if (sorted.length < 2) {
    const nA = data.filter(r => r[targetCol] === 0).length;
    const nB = data.length - nA;
    const n  = data.length;
    return { featureIndex, gini: n > 0 ? 1 - (nA / n) ** 2 - (nB / n) ** 2 : 0, threshold: sorted[0] ?? 0 };
  }

  let bestGini = Infinity, bestThreshold = sorted[0];
  for (let k = 0; k < sorted.length - 1; k++) {
    const t = (sorted[k] + sorted[k + 1]) / 2;
    let lA = 0, lB = 0, rA = 0, rB = 0;
    for (const row of data) {
      if (row[fname] <= t) { row[targetCol] === 0 ? lA++ : lB++; }
      else                 { row[targetCol] === 0 ? rA++ : rB++; }
    }
    const g = weightedGini(lA, lB, rA, rB);
    if (g < bestGini) { bestGini = g; bestThreshold = t; }
  }
  return { featureIndex, gini: bestGini, threshold: +bestThreshold.toFixed(4) };
}

// Recursive CART builder.
// Output shape mirrors the fake buildTreeData so the visualizer works unchanged.
export function buildRealTree(data, features, targetCol, maxDepth, subsetSize, depth = 0) {
  const n      = data.length;
  const classA = data.filter(r => r[targetCol] === 0).length;
  const classB = n - classA;
  const impurity = n > 0 ? 1 - (classA / n) ** 2 - (classB / n) ** 2 : 0;

  // Leaf conditions: max depth, too few samples, or pure node
  if (depth >= maxDepth || n <= 5 || classA === 0 || classB === 0) {
    return { type: "leaf", samples: n, classA, classB, impurity, prediction: classA >= classB ? "A" : "B", depth };
  }

  // Evaluate ALL features (Gini + best threshold for each)
  const allIdx = features.map((_, i) => i);
  const allFeatureEvals = allIdx.map(fi => evalFeatureOnData(data, fi, features, targetCol));

  // Random feature subset
  const candidateIndices = shuffleArray(allIdx).slice(0, subsetSize);

  // Best split within subset
  let bestFeature = candidateIndices[0], bestGini = Infinity, bestThreshold = 0;
  for (const fi of candidateIndices) {
    const ev = allFeatureEvals[fi];
    if (ev.gini < bestGini) { bestGini = ev.gini; bestFeature = fi; bestThreshold = ev.threshold; }
  }

  // Global best (for the "suboptimality" annotation in the feature pool)
  let globalBestIdx = 0, globalBestGini = Infinity;
  for (const ev of allFeatureEvals) {
    if (ev.gini < globalBestGini) { globalBestGini = ev.gini; globalBestIdx = ev.featureIndex; }
  }

  const fname    = features[bestFeature];
  const leftData = data.filter(r => r[fname] <= bestThreshold);
  const rightData = data.filter(r => r[fname] > bestThreshold);

  // Degenerate split — return leaf instead of infinite recursion
  if (leftData.length === 0 || rightData.length === 0) {
    return { type: "leaf", samples: n, classA, classB, impurity, prediction: classA >= classB ? "A" : "B", depth };
  }

  return {
    type: "split",
    featureIndex: bestFeature,
    featureName: features[bestFeature],
    threshold: bestThreshold,
    gini: bestGini,
    samples: n,
    candidateIndices,
    allFeatureEvals,
    globalBestIdx,
    globalBestGini,
    depth,
    left:  buildRealTree(leftData,  features, targetCol, maxDepth, subsetSize, depth + 1),
    right: buildRealTree(rightData, features, targetCol, maxDepth, subsetSize, depth + 1),
  };
}

// Sample n rows with replacement. Returns bootstrapData (array of row objects)
// and oobIndices (original indices NOT drawn).
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

// Run a single row through the tree to get "A" or "B".
export function predictRow(tree, row, features) {
  if (tree.type === "leaf") return tree.prediction;
  const fname = features[tree.featureIndex];
  return row[fname] <= tree.threshold
    ? predictRow(tree.left,  row, features)
    : predictRow(tree.right, row, features);
}

// Compute OOB accuracy: run each OOB row through the tree, compare to true label.
export function computeOOBAccuracy(tree, data, oobIndices, features, targetCol) {
  if (oobIndices.length === 0) return null;
  let correct = 0;
  for (const idx of oobIndices) {
    const pred   = predictRow(tree, data[idx], features);
    const actual = data[idx][targetCol] === 0 ? "A" : "B";
    if (pred === actual) correct++;
  }
  return +(correct / oobIndices.length).toFixed(3);
}
