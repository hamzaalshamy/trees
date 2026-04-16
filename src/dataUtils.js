// ─── CSV / data-processing utilities shared by RandomForestViz and AdaBoostViz ─

export const NA_VALS = new Set(["", "NA", "na", "nan", "NaN", "?", "null", "undefined", "N/A", "n/a"]);

// Turn a raw target value into a display label.
// Numeric values get a "Class " prefix; string values are used as-is (capitalised).
export function formatClassLabel(raw) {
  const s = String(raw).trim();
  if (s === "") return "Unknown";
  const n = Number(s);
  if (!isNaN(n) && s !== "") return `Class ${s}`;
  return s.charAt(0).toUpperCase() + s.slice(1);
}

export function detectNAs(rows, headers) {
  let total = 0;
  const byCol = {};
  headers.forEach(h => { byCol[h] = 0; });
  rows.forEach(r => headers.forEach(h => {
    if (NA_VALS.has(String(r[h] ?? "").trim())) { total++; byCol[h]++; }
  }));
  return { total, byCol };
}

// Stratified random sample without replacement, preserving class proportions.
export function stratifiedSample(rows, targetCol, n) {
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
    const g = [...group];
    for (let i = g.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [g[i], g[j]] = [g[j], g[i]];
    }
    result.push(...g.slice(0, Math.min(take, g.length)));
  }
  for (let i = result.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [result[i], result[j]] = [result[j], result[i]];
  }
  return result;
}

// selectedCols: array of column names the user kept (includes target). null = use all headers.
export function processCSVData(rawRows, headers, targetCol, naStrategy, sampleMode, selectedCols, taskType = "classification") {
  const activeCols = selectedCols
    ? [targetCol, ...selectedCols.filter(h => h !== targetCol)]
    : headers;

  let rows;
  if (naStrategy === "drop") {
    rows = rawRows.filter(r => activeCols.every(h => !NA_VALS.has(String(r[h] ?? "").trim())));
  } else {
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

  const isNum = {};
  featCols.forEach(h => {
    const vals = rows.map(r => String(r[h] ?? "").trim()).filter(v => !NA_VALS.has(v));
    isNum[h] = vals.length > 0 && vals.every(v => v !== "" && !isNaN(parseFloat(v)));
  });

  const catVals = {};
  featCols.filter(h => !isNum[h]).forEach(h => {
    catVals[h] = [...new Set(rows.map(r => String(r[h] ?? "")))].sort();
  });

  const features = [];
  featCols.forEach(h => {
    if (isNum[h]) {
      features.push(h);
    } else {
      catVals[h].slice(1).forEach(v => features.push(`${h}_${v}`));
    }
  });

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

// Auto-detect whether a target column looks like regression (≥20 unique numeric values).
export function detectTaskType(rawRows, targetCol) {
  const vals = rawRows.map(r => String(r[targetCol] ?? "").trim());
  const hasNonNumeric = vals.some(v => v === "" || isNaN(parseFloat(v)));
  if (hasNonNumeric) return "classification";
  return new Set(vals).size >= 20 ? "regression" : "classification";
}
