# 🎓 Enterprise Machine Learning Textbook & Master Class

Welcome to the **Enterprise Machine Learning Master Course**. This textbook-style curriculum covers end-to-end applied machine learning—from foundational data cleaning and feature engineering to mathematical derivations of modern gradient boosting algorithms (XGBoost, LightGBM, CatBoost), regularization theory, dimensionality reduction, probability calibration, and enterprise deployment strategies.

Throughout this course, theoretical concepts are directly anchored to real-world code implementations from the **12 Production ML Pipelines** in this repository (`src/`).

---

## 📖 Curriculum Syllabus & Table of Contents

### 📘 [Chapter 1: Data Cleaning, Imputation & Preprocessing Pipelines](file:///e:/Downloads/ML_only/course/01_data_cleaning_and_preprocessing.md)
- **Missing Data Taxonomy**: Missing Completely at Random (MCAR), Missing at Random (MAR), Missing Not at Random (MNAR).
- **Imputation Strategies**: Univariate (Mean/Median/Mode), Multivariate (KNNImputer, MICE - IterativeImputer).
- **Outlier Detection & Remediation**: IQR Bounds, $Z$-Score filtering, Isolation Forests, Mahalanobis Distance.
- **Data Leakage Prevention**: Enforcing strict train/test boundaries in preprocessing pipelines.
- *Repository Case Anchor*: Preprocessing pipeline in [`src/medical_diagnosis/pipeline.py`](file:///e:/Downloads/ML_only/src/medical_diagnosis/pipeline.py).

---

### 📙 [Chapter 2: Feature Engineering Mastery & Signal Extraction](file:///e:/Downloads/ML_only/course/02_feature_engineering_mastery.md)
- **Categorical Feature Encoding**: One-Hot, Ordinal, Target Encoding (Smoothing & Out-of-Fold Regularization), Frequency Encoding.
- **Numerical Feature Scaling**: StandardScaler ($Z$-score), MinMaxScaler, RobustScaler (IQR quantile scaling).
- **Non-Linear Mathematical Transformations**: $\log(1+x)$, Box-Cox, Yeo-Johnson, Power Transformations for skewed features.
- **Time-Series Feature Engineering**: Autoregressive Lags ($y_{t-k}$), Rolling Moving Averages, Exponential Smoothing, Cyclical Sine/Cosine Encoding.
- *Repository Case Anchor*: Time-Series feature engineering in [`src/demand_forecasting/pipeline.py`](file:///e:/Downloads/ML_only/src/demand_forecasting/pipeline.py).

---

### 📗 [Chapter 3: Bias-Variance Decomposition & Regularization Theory](file:///e:/Downloads/ML_only/course/03_bias_variance_and_regularization.md)
- **Overfitting vs. Underfitting**: Learning curves, capacity, and generalization gap.
- **Mathematical Bias-Variance Decomposition**: Derivation of Expected Generalization Error:
  $$\mathbb{E}[(y - \hat{f}(x))^2] = \text{Bias}[\hat{f}(x)]^2 + \text{Var}[\hat{f}(x)] + \sigma^2$$
- **Regularization Fundamentals**:
  - **L1 Regularization (Lasso)**: Sparse weight vectors & automatic feature selection ($\|w\|_1$).
  - **L2 Regularization (Ridge)**: Weight shrinkage & multicollinearity mitigation ($\|w\|_2^2$).
  - **ElasticNet**: Combining L1 and L2 penalty regions.
- **Geometric Intuition**: Constraint diamond $\|w\|_1 \le C$ vs. ball $\|w\|_2^2 \le C$.
- *Repository Case Anchor*: Scorecard regularization in [`src/credit_risk/pipeline.py`](file:///e:/Downloads/ML_only/src/credit_risk/pipeline.py).

---

### 📕 [Chapter 4: Dimensionality Reduction & PCA Mathematics](file:///e:/Downloads/ML_only/course/04_dimensionality_reduction_pca.md)
- **Curse of Dimensionality**: High-dimensional volume sparsity & distance concentration.
- **Principal Component Analysis (PCA)**: Mathematical derivation via Covariance Matrix Eigen-decomposition and Singular Value Decomposition (SVD):
  $$\Sigma = \frac{1}{N} X^T X = V \Lambda V^T$$
- **Explained Variance Ratio**: Scree plots & selecting optimal component cutoff.
- **TruncatedSVD vs. Standard PCA**: Handling sparse text and rating matrices.
- *Repository Case Anchor*: Unsupervised 2D PCA mapping in [`src/customer_segmentation/pipeline.py`](file:///e:/Downloads/ML_only/src/customer_segmentation/pipeline.py) and matrix factorization in [`src/recommendation/pipeline.py`](file:///e:/Downloads/ML_only/src/recommendation/pipeline.py).

---

### 📓 [Chapter 5: Advanced Tree Ensembles — XGBoost, LightGBM & CatBoost](file:///e:/Downloads/ML_only/course/05_tree_ensembles_xgboost_lightgbm_catboost.md)
- **Decision Tree Foundations**: Information Gain, Entropy, Gini Impurity, Variance Reduction.
- **Ensemble Philosophy**: Bagging (Bootstrap Aggregating) vs. Boosting (Gradient Descent in Function Space).
- **Random Forest**: Subsampling rows & features, variance reduction via tree decorrelation.
- **XGBoost (Extreme Gradient Boosting)**: 2nd-Order Taylor Expansion ($g_i, h_i$), Regularization ($\gamma, \lambda$), Weighted Quantile Sketch.
- **LightGBM**: Gradient-based One-Side Sampling (GOSS), Exclusive Feature Bundling (EFB), Leaf-wise (best-first) tree growth.
- **CatBoost**: Ordered Boosting (target leakage prevention), Oblivious (Symmetric) Trees, automatic categorical combinations.
- **Comparative Decision Matrix**: Benchmarking speed, memory, accuracy, and categorical handling across algorithms.
- *Repository Case Anchor*: Gradient Boosting models in [`src/fraud_detection/pipeline.py`](file:///e:/Downloads/ML_only/src/fraud_detection/pipeline.py) and [`src/customer_churn/pipeline.py`](file:///e:/Downloads/ML_only/src/customer_churn/pipeline.py).

---

### 📘 [Chapter 6: Model Evaluation, Validation & Calibration](file:///e:/Downloads/ML_only/course/06_model_evaluation_and_validation.md)
- **Validation Strategies**: Stratified $K$-Fold CV, Time-Series Expansion Windows (preventing data leakage).
- **Classification Evaluation**: Confusion Matrix, Precision, Recall, $F_1$, $F_\beta$ Score, ROC-AUC, Precision-Recall AUC (for severe class imbalance).
- **Regression Evaluation**: MSE, RMSE, MAE, MAPE, $R^2$, Adjusted $R^2$.
- **Probability Calibration**: Reliability Diagrams, Platt Scaling (Sigmoidal), Isotonic Regression.
- *Repository Case Anchor*: Model evaluation & metrics serialization across all 12 pipelines in [`scripts/train_all.py`](file:///e:/Downloads/ML_only/scripts/train_all.py).

---

### 📙 [Chapter 7: End-to-End Industry Case Studies](file:///e:/Downloads/ML_only/course/07_end_to_end_project_case_studies.md)
- Comprehensive walkthroughs of the 12 production pipelines, dissecting real-world problem formulations, feature choices, model trade-offs, and deployment architectures.

---

## 🎯 How to Use This Course
1. **Linear Learning Path**: Read Chapters 1 through 6 sequentially for a rigorous theoretical foundation.
2. **Applied Reference**: Study Chapter 7 alongside the source code in `src/` to see theoretical principles translated into production Python code.
