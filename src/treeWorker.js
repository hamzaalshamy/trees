// Web Worker: builds each tree independently and posts results back one-by-one.
// Imported with Vite's ?worker syntax: import TreeWorker from "./treeWorker.js?worker"

import { buildRealTree, bootstrapSample, computeOOBAccuracy, computeOOBMetricsRegression } from "./cartAlgorithm.js";

// Duplicate of the flattenNodes helper (can't import from RandomForestViz).
function flattenNodes(node, id = "0") {
  node.id = id;
  const nodes = [node];
  if (node.type === "split") {
    nodes.push(...flattenNodes(node.left,  id + "L"));
    nodes.push(...flattenNodes(node.right, id + "R"));
  }
  return nodes;
}

self.onmessage = ({ data: { data, features, targetCol, maxDepth, subSize, nEstimators, mode } }) => {
  const n = data.length;
  for (let i = 0; i < nEstimators; i++) {
    const { bootstrapData, oobIndices, inBag } = bootstrapSample(data);
    const t = buildRealTree(bootstrapData, features, targetCol, maxDepth, subSize, 0, mode);
    flattenNodes(t);

    let bInfo;
    if (mode === "regression") {
      const oobMetrics = computeOOBMetricsRegression(t, data, oobIndices, features, targetCol);
      bInfo = { inBag, oob: n - inBag, bootstrapN: n, oobR2: oobMetrics?.r2 ?? null, oobRMSE: oobMetrics?.rmse ?? null };
    } else {
      const oobAcc = computeOOBAccuracy(t, data, oobIndices, features, targetCol);
      bInfo = { inBag, oob: n - inBag, bootstrapN: n, oobAccuracy: oobAcc ?? 0 };
    }

    self.postMessage({ type: "tree", idx: i, tree: t, bInfo });
  }
  self.postMessage({ type: "done" });
};
