import Papa from "papaparse";
import rawCSV from "./heart.csv?raw";

const CP_MAP    = { "typical angina": 0, "atypical angina": 1, "non-anginal": 2, "asymptomatic": 3 };
const ECG_MAP   = { "normal": 0, "lv hypertrophy": 1, "st-t abnormality": 2 };
const SLOPE_MAP = { "upsloping": 0, "flat": 1, "downsloping": 2 };
const THAL_MAP  = { "normal": 0, "fixed defect": 1, "reversable defect": 2 };

// Human-readable feature names — these are also the keys used in each data row object,
// so the CART algorithm's row[featureName] lookup works directly with these.
export const HEART_FEATURE_NAMES = [
  "Age",
  "Sex",
  "Chest Pain Type",
  "Resting Blood Pressure",
  "Cholesterol",
  "Fasting Blood Sugar",
  "Resting ECG",
  "Max Heart Rate",
  "Exercise Angina",
  "ST Depression",
  "ST Slope",
  "Major Vessels",
  "Thalassemia",
];

export const HEART_FEATURE_DESCRIPTIONS = {
  "Age":                    "Age in years",
  "Sex":                    "Sex (1 = Male, 0 = Female)",
  "Chest Pain Type":        "0 = typical angina, 1 = atypical angina, 2 = non-anginal, 3 = asymptomatic",
  "Resting Blood Pressure": "Resting blood pressure on admission (mm Hg)",
  "Cholesterol":            "Serum cholesterol (mg/dl)",
  "Fasting Blood Sugar":    "Fasting blood sugar > 120 mg/dl (1 = True, 0 = False)",
  "Resting ECG":            "0 = normal, 1 = LV hypertrophy, 2 = ST-T wave abnormality",
  "Max Heart Rate":         "Maximum heart rate achieved (bpm)",
  "Exercise Angina":        "Exercise-induced angina (1 = Yes, 0 = No)",
  "ST Depression":          "ST depression induced by exercise relative to rest",
  "ST Slope":               "Slope of peak exercise ST segment (0 = up, 1 = flat, 2 = down)",
  "Major Vessels":          "Number of major vessels (0–3) colored by fluoroscopy",
  "Thalassemia":            "0 = normal, 1 = fixed defect, 2 = reversable defect",
};

function processRow(r) {
  const age     = +r.age;
  const sex     = r.sex === "Male" ? 1 : 0;
  const cp      = CP_MAP[r.cp];
  const trestbps= +r.trestbps;
  const chol    = +r.chol;
  const fbs     = r.fbs === "TRUE" ? 1 : 0;
  const restecg = ECG_MAP[r.restecg];
  const thalch  = +r.thalch;
  const exang   = r.exang === "TRUE" ? 1 : 0;
  const oldpeak = +r.oldpeak;
  const slope   = SLOPE_MAP[r.slope];
  const ca      = r.ca === "" || r.ca === undefined ? NaN : +r.ca;
  const thal    = THAL_MAP[r.thal];
  const target  = +r.num > 0 ? "Disease" : "No Disease";

  const vals = [age, sex, cp, trestbps, chol, fbs, restecg, thalch, exang, oldpeak, slope, ca, thal];
  if (vals.some(v => v === undefined || Number.isNaN(v))) return null;

  return {
    "Age":                    age,
    "Sex":                    sex,
    "Chest Pain Type":        cp,
    "Resting Blood Pressure": trestbps,
    "Cholesterol":            chol,
    "Fasting Blood Sugar":    fbs,
    "Resting ECG":            restecg,
    "Max Heart Rate":         thalch,
    "Exercise Angina":        exang,
    "ST Depression":          oldpeak,
    "ST Slope":               slope,
    "Major Vessels":          ca,
    "Thalassemia":            thal,
    target,
  };
}

const { data: rawRows } = Papa.parse(rawCSV, { header: true, skipEmptyLines: true });
const processedRows = rawRows.map(processRow).filter(Boolean);

export const heartData = processedRows;

export const heartMeta = {
  name: "Heart Disease",
  features: HEART_FEATURE_NAMES,
  featureDescriptions: HEART_FEATURE_DESCRIPTIONS,
  targetCol: "target",
  targetLabels: ["No Disease", "Disease"],
  nSamples: processedRows.length,
  nFeatures: HEART_FEATURE_NAMES.length,
  description: `Heart Disease Dataset — ${processedRows.length} samples, ${HEART_FEATURE_NAMES.length} features, binary classification`,
  tooltip: "UCI Heart Disease dataset — predict presence of heart disease from 13 clinical features (age, blood pressure, cholesterol, ECG results, etc.). " +
           `${processedRows.length} patients, binary classification: Disease vs No Disease`,
};
