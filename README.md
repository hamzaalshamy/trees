# Trees

Interactive visualizer for tree-based ML algorithms. Watch decision trees, bagging, and random forests build step by step on real data.

[screenshot of landing page: docs/images/landing.png]

## What it does

Trees lets you watch machine learning algorithms think. Step through tree construction one node at a time, see which features are evaluated, understand why the algorithm picks each split, and watch the ensemble come together.

[screenshot of tree growing with feature pool: docs/images/tree-growing.png]

### Feature selection visualization

At each split, the feature pool shows every feature's Gini impurity. Candidate features light up in orange, the chosen best turns green — and when the true best wasn't in the random subset, it highlights in red so you can see the cost of randomness.

[screenshot of feature pool: docs/images/feature-pool.png]

### Prediction & path tracing

Select any sample from the dataset and watch it travel through each tree in the forest. The path lights up from root to leaf, every tree casts its vote, and the majority determines the prediction.

[screenshot of prediction with path: docs/images/prediction.png]

### Bring your own data

Drop any CSV file. Trees parses it in your browser, lets you pick the target column, handle missing values, select features, and choose between classification and regression. Your data never leaves your browser.

[screenshot of CSV modal: docs/images/csv-upload.png]

## Supported algorithms

- **Decision Tree** — single tree, all features, no ensemble
- **Bagging** — bootstrap ensemble, all features at each split  
- **Random Forest** — bootstrap ensemble with random feature subsampling
- **AdaBoost** — coming soon
- **Gradient Boosting** — *May* come soon, but unlikely

## Features

- Real CART algorithm with Gini impurity (classification) and MSE (regression)
- Binary, multiclass, and regression support
- Step-by-step animation with arrow key controls
- Per-tree state preservation — switch between trees without losing progress
- Bootstrap sampling with OOB accuracy
- Two built-in datasets: Heart Disease (binary) and Music Genres (10-class)
- CSV upload with preprocessing: missing value handling, column selection, one-hot encoding, stratified sampling

## Tech stack

React · Vite · Papaparse · Web Workers · No backend — entirely client-side

## Run locally

npm install
npm run dev

Open http://localhost:5173

## Built by

Hamza Alshamy — hamzaalshamy.github.io