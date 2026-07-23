# 🎓 Enterprise Machine Learning & AI Engineering 17-Chapter Masterclass

Welcome to the **Enterprise Machine Learning & AI Masterclass**. This complete 17-chapter curriculum covers applied machine learning—from foundational data cleaning, feature engineering, and regularization theory to deep neural networks, XGBoost/LightGBM/CatBoost mathematics, explainable AI (SHAP/LIME), hyperparameter optimization (Optuna), MLOps data drift, time-series LSTMs, graph neural networks, reinforcement learning, Transformers, and Retrieval-Augmented Generation (RAG).

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

### 🧠 [Chapter 8: Neural Networks & Deep Learning Foundations](file:///e:/Downloads/ML_only/course/08_neural_networks_and_deep_learning.md)
- **Artificial Neurons & Activation Functions**: Sigmoid, ReLU, GELU, Swish, and Softmax.
- **Backpropagation Mathematics**: Chain-rule gradient derivations ($\frac{\partial \mathcal{L}}{\partial W^{(l)}}$).
- **Modern Optimizers**: SGD + Momentum, RMSprop, Adam, **AdamW** (decoupled weight decay), Cosine Annealing schedulers.
- **Deep Architectures**: CNNs (ResNet residual skip connections) and **Transformers** (Scaled Dot-Product Self-Attention $Q, K, V$).
- *Repository Case Anchor*: Vision inspection in [`src/defect_detection/pipeline.py`](file:///e:/Downloads/ML_only/src/defect_detection/pipeline.py).

---

### ⚙️ [Chapter 9: MLOps, Data Drift & Model Monitoring](file:///e:/Downloads/ML_only/course/09_mlops_monitoring_and_governance.md)
- **MLOps Continuous Lifecycle Loop**: CI/CD/CT pipelines and automated experiment tracking.
- **Data Drift vs. Concept Drift**: Covariate shift $P(X)$ vs conditional target shift $P(Y \mid X)$.
- **Statistical Drift Detection**: Kolmogorov-Smirnov (KS) tests ($p$-value $< 0.05$), Population Stability Index (PSI).
- **Deployment Strategies**: Blue/Green, Canary Rollouts, and Shadow (Silent) Deployments.
- **Production Monitoring**: Prometheus metrics & FastAPI telemetry endpoints.
- *Repository Case Anchor*: Production API infrastructure in [`api/routes.py`](file:///e:/Downloads/ML_only/api/routes.py).

---

### 🔍 [Chapter 10: Explainable AI (XAI) — SHAP & LIME](file:///e:/Downloads/ML_only/course/10_explainable_ai_shap_lime.md)
- **The Explainability Imperative**: Regulatory compliance (GDPR Right to Explanation, FCRA Adverse Action notices).
- **SHAP Mathematics**: Cooperative Game Theory Shapley values ($\phi_i$), efficiency additivity, TreeSHAP.
- **SHAP Visualizations**: Summary plots, Waterfall plots, and Force plots.
- **LIME Framework**: Local surrogate linear models and perturbation sampling.
- *Repository Case Anchor*: Feature attributions in [`src/fraud_detection/pipeline.py`](file:///e:/Downloads/ML_only/src/fraud_detection/pipeline.py) and [`src/credit_risk/pipeline.py`](file:///e:/Downloads/ML_only/src/credit_risk/pipeline.py).

---

### 🎛️ [Chapter 11: Hyperparameter Optimization & Optuna](file:///e:/Downloads/ML_only/course/11_hyperparameter_optimization_optuna.md)
- **Search Space Strategies**: Grid Search vs Random Search vs Bayesian Optimization.
- **Bayesian Mathematics**: Gaussian Processes (GP) and Expected Improvement (EI) acquisition functions.
- **Optuna Framework**: Tree-structured Parzen Estimators (TPE), Hyperband, and Median Pruners.
- *Repository Case Anchor*: Hyperparameter tuning in [`src/house_prices/pipeline.py`](file:///e:/Downloads/ML_only/src/house_prices/pipeline.py).

---

### ⚡ [Chapter 12: Production Serving & Inference Optimization](file:///e:/Downloads/ML_only/course/12_production_serving_and_inference_optimization.md)
- **Model Serialization Standards**: Joblib vs. ONNX (Open Neural Network Exchange) vs. TensorRT.
- **Inference Optimization**: INT8 Quantization, Weight Pruning, zero-copy memory mapping.
- **FastAPI Microservice Optimization**: Async IO, Warm-Start lifespan caching, multi-worker process concurrency.
- *Repository Case Anchor*: Web application serving in [`api/main.py`](file:///e:/Downloads/ML_only/api/main.py) and containerization in [`Dockerfile`](file:///e:/Downloads/ML_only/Dockerfile).

---

### 📈 [Chapter 13: Advanced Time-Series & Sequential Deep Learning](file:///e:/Downloads/ML_only/course/13_advanced_time_series_and_forecasting.md)
- **Stationarity & Statistical Tests**: Stationarity, Differencing, Augmented Dickey-Fuller (ADF) test ($p$-value $< 0.05$).
- **Classical Models**: ARIMA($p, d, q$), SARIMAX seasonal exogenous models.
- **Sequential Neural Networks**: Recurrent Neural Networks (RNNs) and **LSTM** gating mathematics (Forget gate $f_t$, Input gate $i_t$, Cell state $C_t$, Output gate $o_t$).
- **Modern Forecasting Transformers**: Temporal Fusion Transformer (TFT) and N-BEATS.
- *Repository Case Anchor*: Time-series demand forecasting in [`src/demand_forecasting/pipeline.py`](file:///e:/Downloads/ML_only/src/demand_forecasting/pipeline.py).

---

### 🕵️ [Chapter 14: Unsupervised Anomaly Detection & Density Estimation](file:///e:/Downloads/ML_only/course/14_anomaly_detection_and_density_estimation.md)
- **Density Estimation**: Gaussian Mixture Models (GMM) and Expectation-Maximization (EM) algorithm.
- **Spatial Distance Anomalies**: Mahalanobis Distance ($D_M(x) = \sqrt{(x-\mu)^T \Sigma^{-1} (x-\mu)}$), Local Outlier Factor (LOF), One-Class SVM.
- **Deep Anomaly Architecture**: Autoencoder reconstruction error thresholding ($\|x - \hat{x}\|^2$) and Variational Autoencoders (VAE - ELBO loss $\text{KL}(q(z|x) \parallel p(z))$).
- *Repository Case Anchor*: Transaction anomaly detection in [`src/fraud_detection/pipeline.py`](file:///e:/Downloads/ML_only/src/fraud_detection/pipeline.py).

---

### 🕸️ [Chapter 15: Graph Neural Networks (GNNs) & Network Analytics](file:///e:/Downloads/ML_only/course/15_graph_neural_networks_gnn.md)
- **Graph Representations**: Adjacency matrix $A$, Degree matrix $D$, Feature matrix $X$, Graph Laplacian $L = D - A$.
- **Message Passing Framework**: Node neighborhood feature aggregation and updating ($h_v^{(l+1)} = \text{UPDATE}(h_v^{(l)}, m_v^{(l+1)})$).
- **GNN Architectures**: Graph Convolutional Networks (GCN - $\tilde{D}^{-1/2} \tilde{A} \tilde{D}^{-1/2} H^{(l)} W^{(l)}$) and Graph Attention Networks (GAT - dynamic edge attention $\alpha_{ij}$).
- **Enterprise Applications**: Financial fraud ring detection, anti-money laundering (AML), and transaction graphs.
- *Repository Case Anchor*: Network velocity analysis in [`src/fraud_detection/pipeline.py`](file:///e:/Downloads/ML_only/src/fraud_detection/pipeline.py).

---

### 🎮 [Chapter 16: Reinforcement Learning & Sequential Decision Making](file:///e:/Downloads/ML_only/course/16_reinforcement_learning_foundations.md)
- **Markov Decision Processes (MDP)**: States $\mathcal{S}$, Actions $\mathcal{A}$, Transition Probabilities $\mathcal{P}$, Rewards $\mathcal{R}$, Discount Factor $\gamma$.
- **Bellman Equations**: State-value $V^\pi(s)$, Action-value $Q^\pi(s, a)$, Optimality Equations.
- **Algorithms**: Q-Learning Temporal Difference, Deep Q-Networks (DQN - Experience Replay, Target Network $\theta^-$), Proximal Policy Optimization (PPO clipped surrogate), and Direct Preference Optimization (DPO / RLHF).
- *Repository Case Anchor*: Sequential recommendation discovery in [`src/recommendation/pipeline.py`](file:///e:/Downloads/ML_only/src/recommendation/pipeline.py).

---

### 🔤 [Chapter 17: NLP — From Word Embeddings to LLMs & RAG](file:///e:/Downloads/ML_only/course/17_nlp_embeddings_to_llms.md)
- **Dense Word Embeddings**: Word2Vec (Skip-Gram & CBOW negative sampling), GloVe, FastText subword n-grams.
- **Transformer Architecture**: Multi-Head Self-Attention, Sinusoidal & **Rotary Position Embeddings (RoPE)**.
- **Large Language Models (LLMs)**: Causal Decoder-only modeling ($P(w_t \mid w_{<t})$), Parameter-Efficient Fine-Tuning (**LoRA / QLoRA** low-rank matrix decomposition $W_0 + B \cdot A$).
- **Retrieval-Augmented Generation (RAG)**: Chunking, Vector Database indexing, Cosine similarity search, and prompt context augmentation.
- *Repository Case Anchor*: Text classification & key term extraction in [`src/document_classification/pipeline.py`](file:///e:/Downloads/ML_only/src/document_classification/pipeline.py).
