# Chapter 13: Data Versioning, Lineage, and Reproducibility

---

## 1. Big Picture

In modern ML pipelines data is a first‑class artifact. Just as code is version‑controlled, **datasets must be versioned** so you can reproduce experiments, roll back to a known good state, and audit changes over time.

## 2. Intuition

Think of a dataset snapshot as a *commit* in a Git repository. Each commit records the exact table contents, schema, and associated metadata. When a model is trained, you pin the data version to guarantee that the same inputs can be re‑generated later.

## 3. Core Concepts

- **Immutable Snapshots** – Store each ingest batch as a read‑only Parquet/Delta lake snapshot.
- **Datasets as Objects** – Unique identifier (e.g., `dataset_id@2024-09-14T08:12:00Z`).
- **Lineage Graphs** – Track upstream sources (raw logs, external APIs) and downstream consumers (features, models).
- **Metadata Store** – Central catalog (e.g., Hive Metastore, Glue, or a lightweight SQLite) that records schema, checksum, and provenance.
- **Branching & Merging** – Enable experimental feature pipelines without contaminating production data.

## 4. Tooling Landscape
| Tool | Strength |
|------|----------|
| **Delta Lake** | ACID transactions on cloud storage, time‑travel queries. |
| **Apache Iceberg** | Partitioned metadata, hidden tables, schema evolution. |
| **DVC (Data Version Control)** | Git‑like CLI, integrates with remote storage (S3, GCS). |
| **LakeFS** | Git‑style object store interface, powerful branching. |
| **Great Expectations** | Validation coupled with versioned data expectations. |

## 5. Code Sketch (Python + Delta Lake)
```python
from delta import DeltaTable
import pyspark.sql as spark

# Write immutable snapshot
spark_df = spark.createDataFrame(df)
spark_df.write.format("delta").mode("overwrite").save("s3://ml-data/transactions")

# Time‑travel query to retrieve a historic version
historical = DeltaTable.forPath(spark, "s3://ml-data/transactions")\
    .versionAsOf(42)\
    .toDF()

print(f"Rows at version 42: {historical.count()}")
```

## 6. Operational Practices
- **Trigger version bump** on every successful ingestion job.
- **Store checksum** (SHA‑256) of each snapshot; verify on load.
- **Automate lineage capture** using OpenLineage or MLflow Tracking.
- **Retention policy**: keep raw snapshots for 30 days, curated snapshots for 1 year.

## 7. Real‑World Example
A retail analytics team stored daily transaction logs as Delta tables. When a model regression failed in production, they compared the snapshot used during training (`version 102`) with the current snapshot (`version 157`). The diff revealed a schema change (new `promo_code` column) that broke feature engineering, allowing a rapid rollback.

---

**Key Takeaway:** Treat data like code—immutable, versioned, and fully tracked—so every model run is reproducible and auditable.
