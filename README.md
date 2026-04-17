# Trees

Interactive visualizer for tree-based ML algorithms. Watch machine learning algorithms think, step by step.

🌐 [Live Demo](https://YOUR-VERCEL-URL.vercel.app)

---

## The Algorithms

Trees covers Decision Trees, Bagging, Random Forest, and AdaBoost — each visualized from first split to final prediction. Pick an algorithm from the menu to start exploring.

![Menu](docs/images/landing.png)

---

## Watch Trees Grow

Step through tree construction one split at a time, with Gini impurity scores shown for every candidate feature at each node. The feature pool highlights which features were sampled in the random subset, and marks whether the chosen split was the true global best or a random-subset compromise.

<table>
  <tr>
    <td width="50%"><img src="docs/images/rf_split.png" alt="Single split"/></td>
    <td width="50%"><img src="docs/images/rf_tree_full.png" alt="Full tree"/></td>
  </tr>
  <tr>
    <td align="center"><sub>Single split — feature pool, Gini per candidate, chosen split</sub></td>
    <td align="center"><sub>Full tree — complete tree grown to max depth</sub></td>
  </tr>
</table>

---

## Ensemble Predictions

Once the forest is built, pick any sample from the dataset and watch all trees cast their votes. The ensemble's final prediction is shown alongside the true label, and you can trace the exact decision path any individual tree took to reach its leaf.

<table>
  <tr>
    <td width="50%"><img src="docs/images/rf_prediction.png" alt="Ensemble vote"/></td>
    <td width="50%"><img src="docs/images/rf_prediction_path.png" alt="Decision path"/></td>
  </tr>
  <tr>
    <td align="center"><sub>Ensemble vote — all trees weigh in on one sample</sub></td>
    <td align="center"><sub>Decision path — trace one tree's journey to its prediction</sub></td>
  </tr>
</table>

---

## Bring Your Own Data

Trees isn't limited to the built-in datasets — upload any CSV and the visualizer adapts instantly. It auto-detects columns, lets you choose the target variable, select classification or regression, and configure how missing values are handled.

![CSV Upload](docs/images/csv_upload.png)

---

## Built With

- React 19 + Vite 8
- Web Workers for non-blocking tree construction
- Custom CART implementation (Gini impurity, bootstrap sampling, OOB accuracy)
- React Router v7 for multi-algorithm navigation
- All styling inline — no CSS files or UI libraries

---

## Running Locally

```bash
git clone https://github.com/YOUR_USERNAME/trees.git
cd trees
npm install
npm run dev
```
