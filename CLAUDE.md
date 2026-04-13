# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
npm run dev       # Start development server (Vite HMR)
npm run build     # Production build
npm run lint      # ESLint
npm run preview   # Preview production build
```

## Architecture

**Stack:** React 19 + Vite 8, react-router-dom v7. No CSS files — all styling is inline CSS-in-JS. No state management library. Font: JetBrains Mono.

**Routes:**
```
/                → TaxonomyMenu.jsx     — landing page with SVG algorithm taxonomy
/decision-tree   → RandomForestViz.jsx  — mode="decision-tree"
/bagging         → RandomForestViz.jsx  — mode="bagging"
/random-forest   → RandomForestViz.jsx  — mode="random-forest"
/about           → About.jsx
```

**Shared modules:**
```
src/theme.js              — C color palette (single source of truth for all colors)
src/GlobalHeader.jsx      — persistent nav bar rendered at the top of every page
src/cartAlgorithm.js      — real CART implementation, bootstrap sampling, OOB accuracy
src/data/heartDisease.js  — parses heart.csv at build time via Vite ?raw import
src/data/heart.csv        — UCI Heart Disease dataset (303 rows, 13 features)
```

## GlobalHeader

Rendered at the top of every page. Props: `breadcrumb`, `description`, `center`, `right`.
- Left: "Trees.ML" logo — always navigates to `/`
- Right: always shows "About" link; page-specific content via `right` prop (Upload CSV, reset button)
- Sub-row (algorithm pages only): breadcrumb trail + algorithm description line

On algorithm pages, `center` receives the dataset badge with hover tooltip. The `right` prop gets the Upload CSV label/input and the ↩ Heart Disease reset button.

## RandomForestViz

~800-line component. The `mode` prop (`"decision-tree"` | `"bagging"` | `"random-forest"`) locks hyperparameters:
- `decision-tree`: n_estimators=1, max_features="all" (both locked, controls hidden)
- `bagging`: max_features="all" (locked, control hidden)
- `random-forest`: all controls visible

**Real CART algorithm** (`cartAlgorithm.js`):
- `buildRealTree()` — recursive CART with weighted Gini, best-threshold scan, random feature subsets
- `bootstrapSample()` — sampling with replacement, returns bootstrapData + oobIndices
- `computeOOBAccuracy()` — runs OOB rows through the tree
- `predictRow()` — traverses tree with a real data row

**Tree layout** — leaf-count-based (not exponential): each leaf gets `LEAF_SPACING=110px`, node x is centered over its subtree's leaf range.

**Step/phase system** — each split node animates through 3 phases:
- Phase 0: node appears with orange border showing `?` and `n=X` (feature unknown)
- Phase 1: candidate features highlight orange in the Feature Pool panel; node still shows `?`
- Phase 2: winning feature revealed in node; Feature Pool turns green for chosen, red for global best if excluded

Leaf nodes animate through 2 phases: hidden at phase 0, fully revealed at phase 1.

**Key state:** `trees`, `treeStates` (per-tree animation progress keyed by tree index), `curTree`, `bootstrapInfo`, `zoom`/`pan`, `customDataset`, `csvModal`.

**CSV upload flow:**
1. User drops/selects a CSV (or lands from `/` with `location.state.pendingCSV`)
2. `openFile()` parses with PapaParse, sets `csvModal` state
3. `DataModal` component handles target column selection, NA strategy (drop/fill/ignore), sampling (all / 2000 / 1000 rows stratified)
4. `handleDataConfirm()` receives processed data, calls `buildForestWithData()` immediately

**Datasets:** Default is UCI Heart Disease (303 rows, 13 features, binary). `heartMeta.targetLabels = { A: "No Disease", B: "Disease" }`. Custom CSV datasets derive `classLabels` from the first two unique target values via `formatClassLabel()`.

## TaxonomyMenu

SVG taxonomy diagram (`viewBox="0 0 1060 510"`). Node positions defined in `POS` constant. Active nodes: Decision Tree (amber), Bagging, Random Forest (green). Gray nodes: Boosting, AdaBoost, Gradient Boosting (dotted, coming soon).

Clicking a gray node opens a "Coming soon" popup with email capture (state only, no backend).

CSV drop zone on the landing page reads the file and navigates to `/random-forest` with `{ state: { pendingCSV } }` — RandomForestViz picks this up on mount and opens DataModal automatically.

## Interaction model

- `←` / `→` arrow keys step through tree building
- Mouse wheel / trackpad pinch to zoom; drag to pan
- Double-click a tree tab to instantly complete that tree
- Feature Pool panel: orange = candidate subset, green = chosen, red = global best excluded from subset
- Calculations panel shows live Gini math at every step
