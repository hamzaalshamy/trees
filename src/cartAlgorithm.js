// Real CART (Classification And Regression Trees) implementation.
// Fully general: supports any number of classes — binary is just the n=2 case.

function shuffleArray(arr) {
  const a = [...arr];
  for (let i = a.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [a[i], a[j]] = [a[j], a[i]];
  }
  return a;
}

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

// Evaluate one feature: find the threshold that minimises weighted Gini.
// Returns { featureIndex, gini, threshold }.
function evalFeatureOnData(data, featureIndex, features, targetCol) {
  const fname = features[featureIndex];
  const sorted = [...new Set(data.map(r => r[fname]))].sort((a, b) => a - b);

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

// Recursive CART builder. Works for 2 or more classes.
// Leaf shape: { type:"leaf", samples, classCounts, impurity, prediction, depth }
//   prediction = the class label string with the highest count
export function buildRealTree(data, features, targetCol, maxDepth, subsetSize, depth = 0) {
  const n = data.length;
  const classCounts = countClasses(data, targetCol);
  const impurity = giniFromCounts(classCounts, n);
  // Argmax: class with highest count
  const prediction = Object.entries(classCounts).sort((a, b) => b[1] - a[1])[0]?.[0] ?? "?";

  // Stop if max depth, too few samples, or node is already pure
  const nDistinct = Object.values(classCounts).filter(c => c > 0).length;
  if (depth >= maxDepth || n <= 5 || nDistinct <= 1) {
    return { type: "leaf", samples: n, classCounts, impurity, prediction, depth };
  }

  // Evaluate ALL features (Gini + best threshold for each)
  const allIdx = features.map((_, i) => i);
  const allFeatureEvals = allIdx.map(fi => evalFeatureOnData(data, fi, features, targetCol));

  // Random feature subset
  const candidateIndices = shuffleArray(allIdx).slice(0, subsetSize);

  // Best split within the subset
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

  const fname     = features[bestFeature];
  const leftData  = data.filter(r => r[fname] <= bestThreshold);
  const rightData = data.filter(r => r[fname] >  bestThreshold);

  // Degenerate split — return leaf instead of infinite recursion
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
    left:  buildRealTree(leftData,  features, targetCol, maxDepth, subsetSize, depth + 1),
    right: buildRealTree(rightData, features, targetCol, maxDepth, subsetSize, depth + 1),
  };
}

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

// Traverse tree for a single row; returns the predicted class label string.
export function predictRow(tree, row, features) {
  if (tree.type === "leaf") return tree.prediction;
  const fname = features[tree.featureIndex];
  return row[fname] <= tree.threshold
    ? predictRow(tree.left,  row, features)
    : predictRow(tree.right, row, features);
}

// OOB accuracy: compare predicted class label to actual class label in the data.
export function computeOOBAccuracy(tree, data, oobIndices, features, targetCol) {
  if (oobIndices.length === 0) return null;
  let correct = 0;
  for (const idx of oobIndices) {
    const pred   = predictRow(tree, data[idx], features);
    const actual = data[idx][targetCol]; // now a class label string
    if (pred === actual) correct++;
  }
  return +(correct / oobIndices.length).toFixed(3);
}
