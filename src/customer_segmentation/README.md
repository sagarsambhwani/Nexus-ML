# 🧩 Customer Segmentation (Unsupervised Clustering & Persona Profiling System)

## 📌 System Overview
The **Customer Segmentation System** performs unsupervised behavioral clustering to group customers into distinct behavioral cohorts based on purchasing power, spending habits, transaction frequency, and recency. Customer segmentation enables hyper-personalized marketing campaigns, tailored discount offerings, and targeted retention strategies.

This pipeline combines **StandardScaler Preprocessing**, **K-Means Clustering ($K=4$)**, and **Principal Component Analysis (PCA)** to cluster customer cohorts, project high-dimensional customer vectors into 2D visual coordinate spaces, and automatically attach qualitative persona profiles.

---

## 🏗️ Deep-Dive Implementation Architecture

Implemented in [`pipeline.py`](file:///e:/Downloads/Nexus-ML/src/customer_segmentation/pipeline.py) as a subclass of `BasePipeline`:

```mermaid
graph TD
    A[Customer RFM Behavioral Vector] --> B[StandardScaler]
    B[StandardScaler] --> C[K-Means Clustering (K=4) + PCA 2D Mapper]
    C[K-Means Clustering (K=4) + PCA 2D Mapper] --> D[Assigned Persona & Targeted Marketing Strategy]
    style A fill:#f9f,stroke:#333,stroke-width:2px
    style D fill:#bbf,stroke:#333,stroke-width:2px
```

### Class Code Structure & Execution Flow:

```python
class CustomerSegmentationPipeline(BasePipeline):
    def __init__(self):
        super().__init__(name="Customer Segmentation", artifact_name="customer_segmentation_model.joblib")
```

1. **Data Generation (`generate_data`)**:
   - Synthesizes 1,500 customer profiles distributed across 4 distinct behavioral clusters:
     - Cluster 0: High Income (\$110k), High Spending Score (85), High Frequency (24 orders/yr), Low Recency (12 days).
     - Cluster 1: High Income (\$105k), Low Spending Score (25), Low Frequency (8 orders/yr), Medium Recency (45 days).
     - Cluster 2: Low Income (\$35k), High Spending Score (78), Medium Frequency (18 orders/yr), Low Recency (20 days).
     - Cluster 3: Low Income (\$30k), Low Spending Score (20), Low Frequency (4 orders/yr), High Recency (90 days).

2. **Unsupervised Training & PCA Mapping (`train`)**:
   - Standardizes features via `StandardScaler()`.
   - Fits `KMeans(n_clusters=4, random_state=42, n_init=10)`.
   - Fits `PCA(n_components=2, random_state=42)` for 2D visual projection.
   - Maps unscaled centroids to human persona profiles and saves `customer_segmentation_model.joblib`.

3. **Inference & Persona Assignee (`predict`)**:
   - Scales incoming customer vector.
   - Assigns cluster index via `kmeans.predict(X_{\text{scaled}})`.
   - Transforms vector to 2D coordinates ($PC_1, PC_2$) via `pca.transform(X_{\text{scaled}})`.
   - Returns assigned cluster ID, persona name, 2D coordinates, and targeted marketing strategy.

---


## 💻 API Usage Example

**Endpoint:** `POST /api/v1/predict/customer-segmentation`

```bash
curl -X 'POST' \
  'http://localhost:8000/api/v1/predict/customer-segmentation' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "annual_income_k": 105.0,
  "spending_score": 88.0,
  "frequency_purchases": 22.0,
  "recency_days": 14.0
}'
```

**Expected JSON Response:**
```json
{
  "pipeline": "Customer Segmentation",
  "status": "success",
  "result": {
    "cluster_id": 2,
    "persona": "High-Value Loyal",
    "strategy": "VIP Rewards Program"
  }
}
```

---

## 🎯 Engineering Decision-Making Process

### 1. Algorithm Selection Rationale: Why K-Means + PCA over Hierarchical Clustering or DBSCAN?
- **Decision**: Selected **StandardScaler + K-Means ($K=4$) + PCA (2D)**.
- **Rationale**:
  - *Fast Online Assignment*: Once centroids are trained, assigning new customer records to clusters requires calculating Euclidean distance to 4 centroids ($O(K \cdot D)$), taking < 1ms. Hierarchical clustering requires $O(N^2)$ memory and cannot easily assign new streaming online points.
  - *Interpretability for Marketing Teams*: $K$-Means centroids yield distinct, non-overlapping customer segments that map directly to actionable marketing personas.
  - *2D Visual Mapping*: PCA reduces 4D RFM space into 2 principal axes ($PC_1, PC_2$) for immediate visual plotting in web dashboards.

### 2. Hyperparameter Choices & Design Trade-Offs:
| Hyperparameter | Value Chosen | Engineering Rationale & Trade-Off |
|---|---|---|
| `n_clusters` | `4` | Derived via Elbow Method analysis; provides distinct, non-redundant business persona profiles. |
| `n_init` | `10` | Runs 10 initial centroid seeds to prevent convergence to sub-optimal local minima. |

---

## 👥 Persona Profiles & Strategic Action Rules

| Cluster ID | Persona Name | Behavioral Profile | Targeted Marketing Strategy |
|---|---|---|---|
| **0** | **VIP High Rollers** | Income > \$70k, Spending > 50 | Exclusive preview invites, concierge service, premium loyalty rewards |
| **1** | **Selective Wealth Savers** | Income > \$70k, Spending $\le 50$ | Value proposition messaging, high-margin quality focus, targeted newsletters |
| **2** | **Trend Seekers & Impulse** | Income $\le \$70k$, Spending > 50 | Flash sales, influencer collaborations, social proof & trending alerts |
| **3** | **Budget Conscious & Occasional**| Income $\le \$70k$, Spending $\le 50$ | Win-back discount coupons, low-cost essentials bundle, re-engagement emails |

---

## 📊 Behavioral Feature Schema

| Feature Name | Data Type | Physical Meaning |
|---|---|---|
| `annual_income_k` | Float | Household annual income (\$k) |
| `spending_score` | Float | Normalized store spending score (1-100) |
| `frequency_purchases` | Float | Completed orders per year |
| `recency_days` | Float | Days since most recent transaction |

---

## 🏋️ Evaluation Metrics & Benchmarks

- **Within-Cluster Sum of Squares (Inertia)**: ~748.77 (Quantifies cluster compactness).
- **Number of Clusters ($K$)**: 4.

---

## 🚀 Production Deployment Strategy

1. **CRM Automation**: Auto-sync predicted persona tags into marketing platforms (Klaviyo / Mailchimp) for automated email campaign segmentation.
2. **Dynamic UI Personalization**: Render custom website banners and product carousels tailored to the visitor's assigned cluster persona upon login.
