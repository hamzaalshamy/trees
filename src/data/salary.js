import Papa from "papaparse";
import rawCSV from "./Salary_Data.csv?raw";

// Normalize education level strings to a consistent set
const EDU_MAP = {
  "high school":       0,
  "bachelor's":        1,
  "bachelor's degree": 1,
  "master's":          2,
  "master's degree":   2,
  "phd":               3,
  "ph.d":              3,
};

const GENDER_MAP = { "male": 1, "female": 0, "other": 2 };

export const SALARY_FEATURE_NAMES = [
  "Age",
  "Gender",
  "Education Level",
  "Years of Experience",
];

export const SALARY_FEATURE_DESCRIPTIONS = {
  "Age":                  "Age in years",
  "Gender":               "0 = Female, 1 = Male, 2 = Other",
  "Education Level":      "0 = High School, 1 = Bachelor's, 2 = Master's, 3 = PhD",
  "Years of Experience":  "Years of professional experience",
};

function processRow(r) {
  const age    = parseFloat(r["Age"]);
  const gender = GENDER_MAP[(r["Gender"] ?? "").toLowerCase().trim()];
  const edu    = EDU_MAP[(r["Education Level"] ?? "").toLowerCase().trim()];
  const yoe    = parseFloat(r["Years of Experience"]);
  const salary = parseFloat(r["Salary"]);

  if ([age, gender, edu, yoe, salary].some(v => v === undefined || Number.isNaN(v))) return null;

  return {
    "Age":                  age,
    "Gender":               gender,
    "Education Level":      edu,
    "Years of Experience":  yoe,
    target:                 salary,
  };
}

// Seeded Fisher-Yates shuffle for a deterministic 1000-row sample
function seededSample(arr, n, seed = 42) {
  const a = [...arr];
  let s = seed >>> 0;
  const rand = () => {
    s = Math.imul(s ^ (s >>> 16), 0x45d9f3b);
    s = Math.imul(s ^ (s >>> 16), 0x45d9f3b);
    s ^= s >>> 16;
    return (s >>> 0) / 0x100000000;
  };
  for (let i = a.length - 1; i > 0; i--) {
    const j = Math.floor(rand() * (i + 1));
    [a[i], a[j]] = [a[j], a[i]];
  }
  return a.slice(0, n);
}

const { data: rawRows } = Papa.parse(rawCSV, { header: true, skipEmptyLines: true });
const processed = rawRows.map(processRow).filter(Boolean);
export const salaryData = processed.length > 1000 ? seededSample(processed, 1000) : processed;

export const salaryMeta = {
  name: "Salary",
  features: SALARY_FEATURE_NAMES,
  featureDescriptions: SALARY_FEATURE_DESCRIPTIONS,
  targetCol: "target",
  targetLabels: null,
  nSamples: salaryData.length,
  nFeatures: SALARY_FEATURE_NAMES.length,
  description: `Salary Dataset — ${salaryData.length} samples, ${SALARY_FEATURE_NAMES.length} features, regression`,
  tooltip: "Salary dataset — predict annual salary from age, gender, education level, and years of experience. Regression task.",
};
