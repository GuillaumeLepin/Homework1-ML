# Homework 1 - SVM

## 1. Review of SVM Strategies
The provided answer describes two multi-class strategies:
- One-vs-Rest (OvR): one classifier per class against all other classes (`K` classifiers for `K` classes).
- One-vs-One (OvO): one classifier for each pair of classes (`K(K-1)/2` classifiers), final prediction by voting.

It also states:
- `LinearSVC` uses OvR in scikit-learn.
- `SVC` uses OvO internally.
- For digits, both are suitable, with linear OvR being efficient and kernel-based OvO being more flexible.

## 2. Regularization Parameter `C`
The provided answer states:
- `C` controls the trade-off between margin size and classification errors.
- Small `C`: stronger regularization, wider margin, more errors, possible underfitting.
- Large `C`: weaker regularization, narrower margin, fewer training errors, possible overfitting.

Additional parameter noted:
- `gamma` (RBF kernel), controlling boundary flexibility:
  - high `gamma`: more complex/local boundary
  - low `gamma`: smoother/more generalized boundary

## 3. Linear SVM Experiment
Reported setup:
- Digits split into 80% train and 20% validation.
- `C` values tested on a logarithmic scale from `10^-4` to `10^4`.

Reported results:
- Best validation accuracy: `0.9694` (96.94%) at `C = 0.1`.
- For larger `C`, accuracy slightly decreases/stabilizes around `0.947`.
- Reported overall interpretation: strong performance, approximately 97% accuracy, minor confusion among similar digits.

## 4. Kernel SVM (RBF) Experiment
Reported setup:
- Same `C` sweep, using RBF kernel.

Reported results:
- Very small `C`: low validation accuracy (around 20%).
- Accuracy improves as `C` increases.
- Best validation accuracy: `0.9806` (98.06%) at `C = 10`, then stable.

Comparison reported versus linear SVM:
- RBF (`0.9806`) outperforms linear (`0.9694`).
- Interpretation: digits are not perfectly linearly separable; nonlinear boundary helps.
- Explanation references kernel trick for implicit higher-dimensional mapping.
