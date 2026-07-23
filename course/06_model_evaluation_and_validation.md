# 📘 Chapter 6: Model Evaluation, Validation & Calibration

## 6.1 Validation Strategies & Cross-Validation

Evaluating a model on the same data used for training leads to overly optimistic error estimates. Robust cross-validation strategies simulate deployment performance on unseen future data.

```
                           K-Fold Cross Validation (K=5)
Fold 1:  [  TEST  |  TRAIN  |  TRAIN  |  TRAIN  |  TRAIN  ] ──► Score 1
Fold 2:  [  TRAIN |  TEST   |  TRAIN  |  TRAIN  |  TRAIN  ] ──► Score 2
Fold 3:  [  TRAIN |  TRAIN  |  TEST   |  TRAIN  |  TRAIN  ] ──► Score 3
Fold 4:  [  TRAIN |  TRAIN  |  TRAIN  |  TEST   |  TRAIN  ] ──► Score 4
Fold 5:  [  TRAIN |  TRAIN  |  TRAIN  |  TRAIN  |  TEST   ] ──► Score 5
                                                                 │
                                                       Average CV Score
```

### 1. Stratified $K$-Fold Cross Validation
- **When to use**: Imbalanced classification datasets (e.g., fraud detection at 2% positive rate).
- **Mechanism**: Guarantees that each fold contains the exact target class distribution ratio as the full dataset.

### 2. Time-Series Expanding Window Validation
> [!WARNING]
> Standard random $K$-Fold cross-validation leaks future data into past training folds, invalidating time-series evaluation.

```
Iter 1: [ Train: Year 1 ] ──► Test: Year 2
Iter 2: [ Train: Year 1 + Year 2 ] ──► Test: Year 3
Iter 3: [ Train: Year 1 + Year 2 + Year 3 ] ──► Test: Year 4
```

---

## 6.2 Classification Evaluation Metrics

```
                            Confusion Matrix
                           Predicted Negative    Predicted Positive
    Actual Negative (0) [      TN              |        FP (Type I)     ]
    Actual Positive (1) [      FN (Type II)    |        TP              ]
```

### Key Metrics Formulations:
1. **Accuracy**:
   $$\text{Accuracy} = \frac{\text{TP} + \text{TN}}{\text{TP} + \text{TN} + \text{FP} + \text{FN}}$$
   *Fails under severe class imbalance.*

2. **Precision** (Quality of positive predictions):
   $$\text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}}$$

3. **Recall / Sensitivity** (Quantity of positive cases captured):
   $$\text{Recall} = \frac{\text{TP}}{\text{TP} + \text{FN}}$$

4. **$F_1$-Score** (Harmonic mean of Precision and Recall):
   $$F_1 = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}} = \frac{2\text{TP}}{2\text{TP} + \text{FP} + \text{FN}}$$

5. **Receiver Operating Characteristic (ROC-AUC)**:
   Plots True Positive Rate ($\text{TPR} = \text{Recall}$) vs False Positive Rate ($\text{FPR} = \frac{\text{FP}}{\text{FP} + \text{TN}}$) across all decision thresholds.

---

## 6.3 Regression Evaluation Metrics

1. **Mean Squared Error (MSE)**:
   $$\text{MSE} = \frac{1}{N} \sum_{i=1}^N (y_i - \hat{y}_i)^2$$
   *Penalizes large outlier errors heavily due to squaring.*

2. **Root Mean Squared Error (RMSE)**:
   $$\text{RMSE} = \sqrt{\text{MSE}} = \sqrt{\frac{1}{N} \sum_{i=1}^N (y_i - \hat{y}_i)^2}$$
   *Interpretable in the same physical units as target variable $y$.*

3. **Mean Absolute Error (MAE)**:
   $$\text{MAE} = \frac{1}{N} \sum_{i=1}^N |y_i - \hat{y}_i|$$
   *Linear error metric robust to extreme outliers.*

4. **Coefficient of Determination ($R^2$ Score)**:
   $$R^2 = 1 - \frac{\text{SS}_{\text{res}}}{\text{SS}_{\text{tot}}} = 1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}$$
   *Measures the proportion of variance explained by the model.*

---

## 6.4 Probability Calibration

Uncalibrated algorithms (like Naive Bayes or raw SVMs) output non-calibrated probability scores. In clinical risk engines and financial underwriting, a predicted score of $0.70$ must correspond to a true empirical default rate of $70\%$.

```
                      Reliability Diagram (Calibration Curve)
              1.0 ┼                                    ╭── Perfectly Calibrated
                  │                                 ╭──╯
       Empirical  │                              ╭──╯
       Frequency  │                           ╭──╯  (Uncalibrated Model)
                  │                        ╭──┼───●
              0.0 ┼───────────────────────┴───┴───────►
                 0.0                         1.0
                          Predicted Probability
```

### Calibration Methods:
1. **Platt Scaling**: Fits a post-processing sigmoid logistic regression curve on top of raw model margin scores $f(x)$:
   $$P(y=1 \mid f(x)) = \frac{1}{1 + \exp(A \cdot f(x) + B)}$$
2. **Isotonic Regression**: Fits a non-parametric monotonic step function. Requires larger calibration sample sizes ($N > 1,000$).

---

## ⚓ Repository Code Reference
- See [`scripts/train_all.py`](file:///e:/Downloads/ML_only/scripts/train_all.py) for metric calculation and logging across all 12 pipelines.
- See [`tests/test_api.py`](file:///e:/Downloads/ML_only/tests/test_api.py) for integration testing validation patterns.
