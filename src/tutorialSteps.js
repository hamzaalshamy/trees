/**
 * Tour step definitions.
 *
 * Each step:
 *   target      – data-tutorial attribute value to spotlight, or null for a centered card
 *   title       – heading shown in the coachmark card
 *   body        – explanation text
 *   position    – preferred card side: 'top' | 'bottom' | 'left' | 'right' | 'center'
 *   action      – optional fn(rfRef) called when this step becomes active
 *   keyboardStep – if true, the tutorial intercepts → and auto-advances after 600ms
 */
export const TOURS = {
  randomForest: [
    // Step 1 — Welcome
    {
      target: null,
      title: "Welcome to Random Forest",
      body: "It builds many independent decision trees on random subsets of your data and has them vote on the final prediction. Let's walk through it.",
      position: "center",
      action: (rfRef) => rfRef.current?.resetAndSetParams({
        maxDepth: 3,
        featureSubset: "sqrt",
        nEstimators: 10,
        speed: 1,
      }),
    },

    // Step 2 — Hyperparameters
    {
      target: "hyperparams",
      title: "Hyperparameters",
      body: "These control how the forest grows. Max depth limits how deep each tree can go. Max features controls how many candidate features each split considers. Trees sets how many trees to build.",
      position: "bottom",
    },

    // Step 3 — Build one split (keyboard-driven)
    {
      target: "tree-canvas",
      title: "Build one split",
      body: "Press → on your keyboard to grow the first split of the tree.",
      position: "right",
      keyboardStep: true,
    },

    // Step 4 — Calculations
    {
      target: "calculations-panel",
      title: "The math behind the split",
      body: "This panel shows the candidate features for this split, their impurity scores, and which one was chosen as the best.",
      position: "left",
    },

    // Step 5 — Feature pool (explanatory, Next button)
    {
      target: "feature-pool",
      title: "Feature pool",
      body: "Orange shows the candidate features this split considered. Green is the one that was chosen.",
      position: "top",
    },

    // Step 6 — Complete All
    {
      target: "complete-all",
      title: "Build the whole forest",
      body: "Now that you've seen how splits work, let's build all 10 trees at once.",
      position: "bottom",
      action: (rfRef) => rfRef.current?.completeAll(),
    },

    // Step 7 — Tree tabs
    {
      target: "tree-tabs",
      title: "Individual trees",
      body: "Each tab is a different tree. They differ because each trained on a random bootstrap sample with random feature subsets at each split. Click any tab to inspect that tree.",
      position: "bottom",
    },

    // Step 8 — Make a prediction
    {
      target: "sample-selector",
      title: "Make a prediction",
      body: "Pick any sample from the dataset to see how the forest predicts it. Let's pick one at random.",
      position: "top",
      action: (rfRef) => {
        rfRef.current?.scrollToPrediction();
        setTimeout(() => rfRef.current?.selectRandomSample(), 120);
      },
    },

    // Step 9 — Ensemble vote
    {
      target: "ensemble-vote",
      title: "Ensemble vote",
      body: "Each tree casts an independent vote. The ensemble prediction is the majority vote — whichever class gets the most trees wins.",
      position: "top",
    },

    // Step 10 — Accuracy metrics
    {
      target: "accuracy-cards",
      title: "Accuracy metrics",
      body: "Training accuracy measures performance on training data. Out-of-bag accuracy tests each tree on the samples it never saw during training — a built-in validation estimate, no held-out set needed.",
      position: "top",
    },

    // Step 11 — Finish
    {
      target: null,
      title: "You're all set!",
      body: "Try changing hyperparameters and rebuilding, upload your own CSV, or explore the other algorithms from the menu. Happy exploring!",
      position: "center",
    },
  ],
};
