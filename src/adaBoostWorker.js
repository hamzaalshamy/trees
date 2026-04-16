// Web Worker: runs the full AdaBoost training off the main thread.
import { runAdaBoost } from "./adaBoostAlgorithm.js";

self.onmessage = ({ data: { data, features, targetCol, nRounds, maxDepth } }) => {
  const result = runAdaBoost(data, features, targetCol, nRounds, maxDepth);
  self.postMessage(result);
};
