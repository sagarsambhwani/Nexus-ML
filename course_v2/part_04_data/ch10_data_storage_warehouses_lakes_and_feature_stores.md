# Chapter 10: Data Storage, Warehouses, Lakes, and Feature Stores

---

## 1. Big Picture

In production ML systems, training and serving data must remain perfectly synchronized. Feature calculation logic executed during offline model training must match online real-time inference feature logic down to the exact floating-point operation.

This chapter covers enterprise data storage architectures: Data Warehouses (Snowflake, BigQuery), Data Lakes (Delta Lake, Apache Iceberg), and **Feature Stores** (Feast, Hopsworks) that solve the **Training-Serving Skew** problem.

---

## 2. Intuition

- **Data Lake (S3 / Blob Storage)**: A giant storage pantry holding raw, un-organized ingredients in bulk (JSON, CSV, MP4, Parquet).
- **Data Warehouse (Snowflake)**: A highly organized restaurant inventory system with strict SQL tables and fast query indexes.
- **Feature Store**: A dual-database system: an **Offline Store** (Parquet/Delta Lake) optimized for batch historical feature extraction, and an **Online Store** (Redis/DynamoDB) optimized for sub-10ms real-time feature retrieval by primary key.

---

## 3. Visualization

```text
                               ┌────────────────────────────────┐
                               │     FEATURE STORE ARCHITECTURE  │
                               └────────────────────────────────┘
                                                │
                 ┌──────────────────────────────┴──────────────────────────────┐
                 ▼                                                             ▼
   OFFLINE STORE (Delta Lake / S3)                           ONLINE STORE (Redis / In-Memory)
   - Batched Historical Point-in-time joins                    - Sub-10ms lookup by Primary Key (user_id)
   - Used for Model Training & Backtesting                    - Used for Real-Time Inference Microservices
```

---

## 4. Mathematics

### Point-In-Time Correctness (Time-Travel Join)
To prevent data leakage when assembling training features, a feature store performs a **As-Of Join**.

Given entity observation events $E = \{(e_i, t_i)\}$ and feature update stream $F = \{(v_j, t_j)\}$:

$$\text{Feature}(e_i, t_i) = v_{k^*} \quad \text{where } k^* = \arg\max_j \{ t_j \mid t_j \le t_i \}$$

This guarantees that feature values $v_j$ updated *after* event time $t_i$ are never leaked into historical training sets!

---

## 5. Python (From Scratch Feature Store & Time-Travel Join)

Implementing a zero-leakage Time-Travel As-Of Join in pure Python & Polars:

```python
import polars as pl
from datetime import datetime

# 1. Historical Observation Events (Target labels)
events_df = pl.DataFrame({
    "user_id": [101, 102, 101],
    "event_timestamp": [
        datetime(2025, 5, 1, 12, 0),
        datetime(2025, 5, 1, 14, 0),
        datetime(2025, 5, 2, 10, 0)
    ],
    "label_churn": [0, 1, 1]
})

# 2. Feature Storage Stream (Features updated over time)
features_df = pl.DataFrame({
    "user_id": [101, 101, 102, 101],
    "feature_timestamp": [
        datetime(2025, 4, 30, 9, 0),   # Value 1 for user 101
        datetime(2025, 5, 1, 13, 0),   # Value 2 for user 101 (AFTER event 1!)
        datetime(2025, 5, 1, 10, 0),   # Value 1 for user 102
        datetime(2025, 5, 2, 8, 0)    # Value 3 for user 101
    ],
    "credit_score": [700, 750, 650, 710]
}).sort("feature_timestamp")

# 3. Perform As-Of Time-Travel Join (Zero Leakage)
joined_df = events_df.sort("event_timestamp").join_asof(
    features_df,
    left_on="event_timestamp",
    right_on="feature_timestamp",
    by="user_id",
    strategy="backward"  # Look backwards in time only
)

print("Point-In-Time Joined Training Dataset:")
print(joined_df)
```

---

## 6. Production Feature Store Configuration (Feast YAML)

```yaml
# feature_store.yaml
project: churn_feature_repo
registry: data/registry.pb
provider: local

offline_store:
  type: file

online_store:
  type: redis
  connection_string: "localhost:6379"
```

---

## 7. Under the Hood

- Polars `join_asof` executes a binary search over sorted timestamp arrays in $O(N \log M)$ time, avoiding cartesian products ($O(N \cdot M)$).
- Feast synchronizes features from Offline storage to Online Redis using materialized batch jobs (`feast materialize <start_date> <end_date>`).

---

## 8. Engineering Perspective

- **Training-Serving Skew**: Occurs when offline feature preprocessing (e.g. Python script) differs from online serving logic (e.g. Java API). Using a centralized feature store eliminates skew by reusing exact feature definitions.

---

## 9. Common Mistakes

1. **Future Data Leakage in Joins**: Joining offline features using simple `user_id` equality without temporal filtering leaks future data into past training samples.
2. **High Online Latency**: Querying SQL relational tables with complex JOINs during real-time API prediction instead of key-value stores (Redis/DynamoDB).

---

## 10. Interview Questions

### Q1: What is Training-Serving Skew and how do Feature Stores prevent it?
**Answer**: Training-serving skew is the discrepancy between model performance during offline training and real-time production serving. It is caused by differences in feature transformation code, data leakage, or temporal mismatches. Feature stores prevent it by using single feature definitions for both batch feature calculation and online key-value retrieval.

---

## 11. Exercises

1. **Coding**: Build a mini Redis-backed online feature store in Python supporting `.get_online_features(entity_keys)`.
2. **Architecture**: Draw an enterprise data lakehouse architecture showing Delta Lake, Spark, Feast, and FastAPI.

---

## 12. Mini Project: In-Memory Feature Store Service

Write a Python microservice `feature_server.py` using FastAPI and Redis that exposes endpoints `/push_features` and `/get_features` with sub-5ms SLA.

---

## 13. Capstone Integration

Connects to `src/customer_churn/pipeline.py` and `src/recommendation/pipeline.py` for online feature lookups and point-in-time dataset assembly.
