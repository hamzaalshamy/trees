import { useState, useEffect, useMemo } from "react";
import { C } from "./theme";
import { processCSVData, NA_VALS, detectTaskType } from "./dataUtils";
import { tooltipPosition } from "./tooltipUtils";

export default function DataModal({ modal, onUpdate, onConfirm, onCancel }) {
  const { fileName, rawRows, headers, naStats, selectedTarget, naStrategy, sampleMode, selectedColumns, warning, taskType = "classification" } = modal;
  const includedCols = selectedColumns ?? headers;
  const [previewOpen, setPreviewOpen] = useState(true);

  const toggleCol = (h) => {
    if (h === selectedTarget) return;
    const next = includedCols.includes(h)
      ? includedCols.filter(c => c !== h)
      : [...includedCols, h];
    if (!next.includes(selectedTarget)) next.push(selectedTarget);
    onUpdate({ selectedColumns: next });
  };

  const handleConfirm = () => {
    const result = processCSVData(rawRows, headers, selectedTarget, naStrategy, sampleMode ?? rawRows.length, includedCols, taskType);
    if (!result) return;
    onConfirm(result.data, result.features, result.targetCol,
              fileName.replace(".csv", ""), result.totalRows, result.sampledRows, result.classLabels, taskType, selectedTarget);
  };

  const featureCols  = headers.filter(h => h !== selectedTarget);
  const checkedCount = featureCols.filter(h => includedCols.includes(h)).length;

  // Inject slider CSS once
  useEffect(() => {
    const id = "sampling-slider-style";
    if (document.getElementById(id)) return;
    const s = document.createElement("style");
    s.id = id;
    s.textContent = `
      input.sampling-slider {
        -webkit-appearance: none; appearance: none;
        outline: none; cursor: pointer;
        background: transparent; height: 24px; width: 100%;
      }
      input.sampling-slider::-webkit-slider-runnable-track {
        height: 4px; background: transparent; border-radius: 2px;
      }
      input.sampling-slider::-moz-range-track {
        height: 4px; background: transparent; border-radius: 2px;
      }
      input.sampling-slider::-webkit-slider-thumb {
        -webkit-appearance: none;
        width: 18px; height: 18px; border-radius: 50%;
        background: #f59e0b; cursor: pointer;
        margin-top: -7px; border: none;
        box-shadow: 0 1px 4px rgba(0,0,0,0.5);
        transition: box-shadow 0.15s ease, width 0.15s ease, height 0.15s ease, margin-top 0.15s ease;
      }
      input.sampling-slider:hover::-webkit-slider-thumb {
        width: 20px; height: 20px; margin-top: -8px;
        box-shadow: 0 1px 8px rgba(0,0,0,0.5), 0 0 0 5px rgba(245,158,11,0.18);
      }
      input.sampling-slider::-moz-range-thumb {
        width: 18px; height: 18px; border-radius: 50%;
        background: #f59e0b; cursor: pointer; border: none;
        box-shadow: 0 1px 4px rgba(0,0,0,0.5);
      }
    `;
    document.head.appendChild(s);
    return () => document.getElementById(id)?.remove();
  }, []);

  // ── Per-column stats for header tooltips ────────────────────────────────────
  const colStats = useMemo(() => {
    const stats = {};
    for (const h of headers) {
      const nonNA  = rawRows
        .map(r => String(r[h] ?? "").trim())
        .filter(v => !NA_VALS.has(v));
      const unique = new Set(nonNA).size;
      stats[h] = { unique, missing: naStats.byCol[h] ?? 0 };
    }
    return stats;
  }, [rawRows, headers, naStats]);

  const [colTooltip, setColTooltip] = useState(null); // { col, x, y }

  // ── Shared style tokens ──────────────────────────────────────────────────────
  const sectionLabel = {
    fontSize: 10, color: C.text, fontWeight: 600,
    display: "block", marginBottom: 8,
  };
  const segTrack = {
    display: "flex", background: "rgba(255,255,255,0.05)",
    borderRadius: 100, padding: 3,
  };
  const segBtn = (active) => ({
    flex: 1, padding: "7px 12px", borderRadius: 100, border: "none",
    cursor: "pointer", fontSize: 10,
    fontWeight: active ? 700 : 400,
    background: active ? C.accent : "transparent",
    color: active ? "#000" : C.dim,
    transition: "background 0.2s ease-out, color 0.2s ease-out",
  });

  return (
    <div style={{
      position: "fixed", inset: 0, background: "rgba(4,8,18,0.82)", zIndex: 100,
      display: "flex", alignItems: "center", justifyContent: "center",
      backdropFilter: "blur(8px)",
    }}>
      <div style={{
        background: "#12192b",
        borderRadius: 20,
        boxShadow: "0 8px 32px rgba(0,0,0,0.55), 0 0 0 1px rgba(255,255,255,0.06)",
        padding: "28px 28px 24px",
        maxWidth: 500, width: "90vw",
        color: C.text,
        maxHeight: "88vh", overflowY: "auto",
        scrollbarWidth: "thin", scrollbarColor: `${C.border} transparent`,
      }}>

        {/* Title */}
        <div style={{ fontSize: 14, fontWeight: 700, color: C.text, marginBottom: 4 }}>
          Configure dataset
        </div>
        <div style={{ fontSize: 10, color: C.dim, marginBottom: 22, lineHeight: 1.7 }}>
          <span style={{ color: C.accent, fontWeight: 600 }}>{fileName}</span>
          {"  ·  "}{rawRows.length.toLocaleString()} rows · {headers.length} columns
          {naStats.total > 0 && <span style={{ color: C.orange, marginLeft: 8 }}>⚠ {naStats.total} missing</span>}
          {warning && <div style={{ color: C.red, marginTop: 3, fontSize: 9 }}>⚠ {warning}</div>}
        </div>

        {/* ── Preview ────────────────────────────────────────────────────────── */}
        <div style={{ marginBottom: 22 }}>
          <button
            onClick={() => setPreviewOpen(o => !o)}
            style={{
              display: "flex", alignItems: "center", gap: 7,
              width: "100%", cursor: "pointer",
              padding: "8px 12px", borderRadius: 9,
              background: previewOpen ? "rgba(245,158,11,0.08)" : "rgba(255,255,255,0.04)",
              border: "none",
              boxShadow: previewOpen
                ? `inset 0 0 0 1px ${C.accent}55`
                : "inset 0 0 0 1px rgba(255,255,255,0.09)",
              fontSize: 10, fontWeight: 500,
              color: previewOpen ? C.accent : C.dim,
              transition: "background 0.15s, box-shadow 0.15s, color 0.15s",
            }}
            onMouseEnter={e => {
              if (!previewOpen) {
                e.currentTarget.style.background = "rgba(255,255,255,0.07)";
                e.currentTarget.style.color = C.text;
              }
            }}
            onMouseLeave={e => {
              e.currentTarget.style.background = previewOpen ? "rgba(245,158,11,0.08)" : "rgba(255,255,255,0.04)";
              e.currentTarget.style.color = previewOpen ? C.accent : C.dim;
            }}
          >
            <span style={{
              fontSize: 10, lineHeight: 1,
              display: "inline-block",
              transition: "transform 0.2s ease-out",
              transform: previewOpen ? "rotate(90deg)" : "rotate(0deg)",
            }}>▶</span>
            Preview &amp; select columns (first 100 rows)
          </button>
          {previewOpen && (
            <>
              <div style={{
                marginTop: 10, overflowX: "auto", overflowY: "auto", height: 290,
                borderRadius: 12,
                background: "#0a0f1a",
                boxShadow: "0 0 0 1px rgba(255,255,255,0.06)",
                scrollbarWidth: "thin", scrollbarColor: `${C.border} transparent`,
              }}>
                <table style={{
                  borderCollapse: "collapse", fontSize: 8.5,
                  fontFamily: "'JetBrains Mono',monospace",
                  whiteSpace: "nowrap", width: "100%",
                  fontVariantNumeric: "tabular-nums",
                }}>
                  <thead>
                    <tr>
                      {headers.map(h => {
                        const isTarget  = h === selectedTarget;
                        const isChecked = isTarget || includedCols.includes(h);
                        return (
                          <th
                            key={h}
                            onClick={() => !isTarget && toggleCol(h)}
                            onMouseEnter={e => {
                              const pos = tooltipPosition(e.currentTarget.getBoundingClientRect(), { prefer: 'below', gap: 4, width: 160, height: 68 });
                              setColTooltip({ col: h, ...pos });
                            }}
                            onMouseLeave={() => setColTooltip(null)}
                            style={{
                              padding: "7px 10px", textAlign: "left",
                              background: "#0a0f1a",
                              position: "sticky", top: 0,
                              boxShadow: "0 1px 0 rgba(255,255,255,0.08)",
                              whiteSpace: "nowrap",
                              cursor: isTarget ? "default" : "pointer",
                              opacity: isChecked ? 1 : 0.35,
                              transition: "opacity 0.15s",
                              userSelect: "none",
                            }}
                          >
                            <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
                              <div style={{
                                width: 13, height: 13, borderRadius: 3, flexShrink: 0,
                                background: isChecked ? C.accent : "transparent",
                                boxShadow: isChecked ? "none" : "0 0 0 1.5px rgba(255,255,255,0.25)",
                                display: "flex", alignItems: "center", justifyContent: "center",
                                transition: "background 0.15s, box-shadow 0.15s",
                                opacity: isTarget ? 0.5 : 1,
                              }}>
                                {isChecked && (
                                  <svg width="8" height="6" viewBox="0 0 8 6" fill="none">
                                    <path d="M1 3L3 5L7 1" stroke="#000" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/>
                                  </svg>
                                )}
                              </div>
                              <div style={{ display: "flex", flexDirection: "column", gap: 1 }}>
                                <div style={{ display: "flex", alignItems: "center", gap: 4 }}>
                                  <span style={{
                                    fontWeight: 600, fontSize: 8.5,
                                    color: isTarget ? C.accent : isChecked ? C.dim : C.dimmer,
                                  }}>
                                    {h}
                                  </span>
                                  {isTarget && (
                                    <span style={{ fontSize: 7, color: C.accent, opacity: 0.65, fontWeight: 400 }}>target</span>
                                  )}
                                </div>
                                {(naStats.byCol[h] ?? 0) > 0 && (
                                  <span style={{ fontSize: 7, color: C.orange, opacity: 0.8, fontWeight: 400 }}>
                                    {naStats.byCol[h]} missing
                                  </span>
                                )}
                              </div>
                            </div>
                          </th>
                        );
                      })}
                    </tr>
                  </thead>
                  <tbody>
                    {rawRows.slice(0, 100).map((row, ri) => (
                      <tr key={ri} style={{ background: ri % 2 === 1 ? "rgba(255,255,255,0.025)" : "transparent" }}>
                        {headers.map(h => {
                          const isTarget  = h === selectedTarget;
                          const isChecked = isTarget || includedCols.includes(h);
                          return (
                            <td key={h} style={{
                              padding: "3.5px 10px",
                              color: isTarget ? "#94a3b8" : C.dimmer,
                              fontWeight: isTarget ? 500 : 400,
                              borderBottom: "1px solid rgba(255,255,255,0.025)",
                              opacity: isChecked ? 1 : 0.25,
                              transition: "opacity 0.15s",
                            }}>
                              {String(row[h] ?? "—")}
                            </td>
                          );
                        })}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              {/* Feature selection summary */}
              <div style={{
                marginTop: 7, fontSize: 9, color: C.dimmer,
                display: "flex", alignItems: "center", gap: 6,
              }}>
                <span style={{ color: checkedCount < featureCols.length ? C.accent : C.dimmer, fontWeight: checkedCount < featureCols.length ? 600 : 400 }}>
                  {checkedCount}/{featureCols.length} features selected
                </span>
                {checkedCount < featureCols.length && (
                  <span>· {featureCols.length - checkedCount} excluded</span>
                )}
                <button
                  onClick={() => onUpdate({ selectedColumns: headers })}
                  style={{
                    marginLeft: "auto", background: "none", border: "none",
                    cursor: "pointer", fontSize: 8.5, color: C.dimmer,
                    padding: 0, transition: "color 0.12s",
                  }}
                  onMouseEnter={e => { e.currentTarget.style.color = C.text; }}
                  onMouseLeave={e => { e.currentTarget.style.color = C.dimmer; }}
                >
                  select all
                </button>
              </div>
            </>
          )}
        </div>

        {/* ── Target column ──────────────────────────────────────────────────── */}
        <div style={{ marginBottom: 22 }}>
          <span style={sectionLabel}>Target column</span>
          <div style={{ position: "relative" }}>
            <select
              value={selectedTarget}
              onChange={e => {
                const newTarget = e.target.value;
                const detected  = detectTaskType(rawRows, newTarget);
                onUpdate({ selectedTarget: newTarget, selectedColumns: headers, taskType: detected });
              }}
              onFocus={e => { e.target.style.boxShadow = `0 0 0 2px ${C.accent}44`; }}
              onBlur={e => { e.target.style.boxShadow = "none"; }}
              style={{
                width: "100%", padding: "9px 32px 9px 12px",
                background: "rgba(255,255,255,0.05)", borderRadius: 10,
                border: "none", color: C.text, fontSize: 11,
                fontFamily: "'JetBrains Mono',monospace",
                cursor: "pointer", outline: "none",
                appearance: "none", WebkitAppearance: "none",
                transition: "box-shadow 0.15s",
              }}>
              {headers.map(h => <option key={h} value={h}>{h}</option>)}
            </select>
            <span style={{
              position: "absolute", right: 12, top: "50%", transform: "translateY(-50%)",
              pointerEvents: "none", color: C.dimmer, fontSize: 10, lineHeight: 1,
            }}>▾</span>
          </div>
        </div>

        {/* ── Task type ──────────────────────────────────────────────────────── */}
        <div style={{ marginBottom: 22 }}>
          <span style={sectionLabel}>Task type</span>
          <div style={segTrack}>
            {["classification", "regression"].map(t => (
              <button key={t} onClick={() => onUpdate({ taskType: t })} style={segBtn(taskType === t)}>
                {t === "classification" ? "Classification" : "Regression"}
              </button>
            ))}
          </div>
          <div style={{ fontSize: 9, color: C.dimmer, marginTop: 7 }}>
            {taskType === "regression"
              ? "Predicts a continuous numeric value · MSE split criterion"
              : "Predicts a class label · Gini impurity split criterion"}
          </div>
        </div>

        {/* ── Missing values ─────────────────────────────────────────────────── */}
        {naStats.total > 0 && (
          <div style={{ marginBottom: 22 }}>
            <span style={sectionLabel}>Handle missing values</span>
            <div style={segTrack}>
              {[
                { key: "drop", label: "Drop rows" },
                { key: "median", label: "Fill with median" },
              ].map(({ key, label }) => (
                <button key={key} onClick={() => onUpdate({ naStrategy: key })} style={segBtn(naStrategy === key)}>
                  {label}
                </button>
              ))}
            </div>
          </div>
        )}

        {/* ── Sampling ───────────────────────────────────────────────────────── */}
        {rawRows.length > 100 && (() => {
          const sv        = typeof sampleMode === "number" ? sampleMode : Math.min(1000, rawRows.length);
          const capVal    = Math.min(rawRows.length, 2000);
          const atCap     = sv >= rawRows.length && rawRows.length < 2000;
          const valuePct  = Math.min((sv - 100) / 1900 * 100, 100);
          const capPct    = Math.min((capVal - 100) / 1900 * 100, 100);
          const recommPct = (1000 - 100) / 1900 * 100;
          const showRecomm = rawRows.length >= 1000;

          const trackGrad = rawRows.length < 2000
            ? `linear-gradient(to right, ${C.accent} 0% ${valuePct}%, rgba(255,255,255,0.09) ${valuePct}% ${capPct}%, rgba(255,255,255,0.03) ${capPct}% 100%)`
            : `linear-gradient(to right, ${C.accent} 0% ${valuePct}%, rgba(255,255,255,0.09) ${valuePct}% 100%)`;

          return (
            <div style={{ marginBottom: 22 }}>
              <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 12 }}>
                <span style={sectionLabel}>Sampling</span>
                <span style={{ fontSize: 13, lineHeight: 1 }}>
                  <span style={{ fontWeight: 600, color: C.accent }}>{sv.toLocaleString()}</span>
                  <span style={{ fontSize: 10, color: C.dim, marginLeft: 4 }}>rows</span>
                </span>
              </div>
              <div style={{ position: "relative", height: 24, marginBottom: 0 }}>
                <div style={{
                  position: "absolute", left: 0, right: 0, top: "50%",
                  transform: "translateY(-50%)", height: 4, borderRadius: 2,
                  background: trackGrad, pointerEvents: "none",
                }} />
                {rawRows.length < 2000 && (
                  <div style={{
                    position: "absolute", top: "50%", left: `${capPct}%`,
                    transform: "translate(-50%, -50%)",
                    width: 1.5, height: 8, borderRadius: 1,
                    background: C.dimmer, pointerEvents: "none", opacity: 0.6,
                  }} />
                )}
                <input
                  type="range"
                  className="sampling-slider"
                  min={100} max={2000} step={100}
                  value={sv}
                  onChange={e => {
                    const v = Math.min(+e.target.value, rawRows.length);
                    onUpdate({ sampleMode: v });
                  }}
                />
              </div>
              <div style={{ position: "relative", height: 20, marginBottom: 6 }}>
                <span style={{ position: "absolute", left: 0, fontSize: 7.5, color: C.text, top: 4 }}>100</span>
                <span style={{ position: "absolute", right: 0, fontSize: 7.5, color: C.text, top: 4 }}>2,000</span>
                {showRecomm && (
                  <div style={{
                    position: "absolute", top: 0, left: `${recommPct}%`,
                    transform: "translateX(-50%)",
                    display: "flex", flexDirection: "column", alignItems: "center",
                    pointerEvents: "none",
                  }}>
                    <div style={{ width: 1, height: 5, background: C.dim }} />
                    <span style={{ fontSize: 7, color: C.dim, whiteSpace: "nowrap", marginTop: 1 }}>recommended</span>
                  </div>
                )}
              </div>
              {atCap ? (
                <div style={{ fontSize: 9, color: C.orange }}>
                  Dataset only has {rawRows.length.toLocaleString()} rows — using all data
                </div>
              ) : (
                <div style={{ fontSize: 9, color: C.dimmer }}>
                  Stratified sampling preserves class proportions
                </div>
              )}
            </div>
          );
        })()}

        {/* Privacy note */}
        <div style={{ fontSize: 8.5, color: C.dimmer, marginBottom: 20, textAlign: "center", letterSpacing: "0.02em" }}>
          🔒 Your data never leaves your browser
        </div>

        {/* Actions */}
        <div style={{ display: "flex", gap: 8 }}>
          <button onClick={onCancel}
            onMouseEnter={e => { e.currentTarget.style.color = C.text; e.currentTarget.style.background = "rgba(255,255,255,0.06)"; }}
            onMouseLeave={e => { e.currentTarget.style.color = C.dim; e.currentTarget.style.background = "transparent"; }}
            style={{
              flex: 1, padding: "10px", borderRadius: 12, border: "none",
              background: "transparent", color: C.dim, fontSize: 11,
              fontFamily: "inherit", cursor: "pointer",
              transition: "color 0.15s, background 0.15s",
            }}>Cancel</button>
          <button onClick={handleConfirm}
            onMouseEnter={e => { e.currentTarget.style.transform = "scale(1.02)"; }}
            onMouseLeave={e => { e.currentTarget.style.transform = "scale(1)"; }}
            style={{
              flex: 2, padding: "10px", borderRadius: 12, border: "none",
              background: `linear-gradient(135deg,${C.accent},#d97706)`,
              color: "#000", fontSize: 11, fontFamily: "inherit",
              cursor: "pointer", fontWeight: 700,
              transition: "transform 0.15s ease-out",
            }}>Confirm &amp; Build</button>
        </div>
      </div>

      {/* ── Column header tooltip ──────────────────────────────────────────── */}
      {colTooltip && colStats[colTooltip.col] && (() => {
        const s = colStats[colTooltip.col];
        return (
          <div style={{
            position: "fixed",
            left: colTooltip.left,
            top: colTooltip.top,
            zIndex: 9999,
            background: "#1a2235",
            borderRadius: 8,
            padding: "8px 11px",
            boxShadow: "0 8px 24px rgba(0,0,0,0.6), inset 0 0 0 1px rgba(255,255,255,0.08)",
            pointerEvents: "none",
          }}>
            <div style={{ display: "flex", flexDirection: "column", gap: 3, fontSize: 9, color: C.dim }}>
              <div style={{ display: "flex", justifyContent: "space-between", gap: 16 }}>
                <span>Unique values</span>
                <span style={{ color: C.text, fontFamily: "'JetBrains Mono',monospace" }}>{s.unique}</span>
              </div>
              <div style={{ display: "flex", justifyContent: "space-between", gap: 16 }}>
                <span>Missing</span>
                <span style={{ color: s.missing > 0 ? C.orange : C.dim, fontFamily: "'JetBrains Mono',monospace" }}>
                  {s.missing}
                </span>
              </div>
            </div>
          </div>
        );
      })()}
    </div>
  );
}
