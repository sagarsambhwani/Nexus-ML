# Chapter 06: SQL, Parquet, Columnar Formats, and Production APIs

---

## 1. Big Picture

In machine learning engineering, data does not arrive as pristine CSV files. Production data resides in relational databases (PostgreSQL, MySQL), analytical data warehouses (Snowflake, BigQuery, DuckDB), columnar file lakes (Apache Parquet, ORC, Feather), and REST microservices.

An ML Engineer must write optimized SQL queries, leverage columnar storage for 10x-100x IO speedups, and interface with FastAPI endpoints for production data ingestion and model serving.

---

## 2. Intuition

- **CSV / Row-Based Storage**: Reading a book by reading word #1 from page 1, word #1 from page 2, word #1 from page 3... If you only want to analyze item prices, you still have to read names, addresses, timestamps, and descriptions for every row!
- **Parquet / Columnar Storage**: Arranging the book so all item prices are stored together in continuous memory. If you ask for prices, the disk head reads a single continuous block instantly, skipping 95% of unneeded data.

---

## 3. Visualization

```text
Row-Oriented Format (CSV / Postgres Table):
  Row 1: [ID1 | Name1 | Age1 | Salary1]
  Row 2: [ID2 | Name2 | Age2 | Salary2]
  (Reading 'Salary' requires scanning ALL columns across disk)

Columnar Format (Parquet / Arrow):
  ID Block:     [ID1 | ID2 | ID3 | ...]
  Name Block:   [Name1 | Name2 | ...]
  Age Block:    [Age1 | Age2 | ...]
  Salary Block: [Salary1 | Salary2 | ...]
  (Reading 'Salary' scans ONLY the Salary block!)
```

---

## 4. Mathematics

### Parquet Compression Ratio Formula
The space compression ratio $R$ achieved by columnar Dictionary Encoding + Snappy/ZSTD compression over raw text CSV is:

$$R = \frac{\text{Size}_{\text{CSV}}}{\text{Size}_{\text{Parquet}}} = \frac{\sum_{i=1}^N \sum_{j=1}^M \text{len}(\text{str}(x_{i,j}))}{\sum_{j=1}^M \left( U_j \cdot \text{len}(v_j) + N \cdot \lceil \log_2(U_j) \rceil / 8 \right) + \text{MetadataBytes}}$$

Where $U_j$ is the count of unique values in column $j$ ($U_j \ll N$ for categorical columns).

---

## 5. Python (DuckDB SQL + Parquet Benchmark)

Using **DuckDB** for in-process SQL analytical queries directly on Parquet files:

```python
import duckdb
import polars as pl
import numpy as np

# 1. Generate 1M Row Synthetic Dataset & Save as Parquet
N = 1_000_000
df = pl.DataFrame({
    'timestamp': pl.date_range(start=pl.datetime(2025, 1, 1), end=pl.datetime(2025, 12, 31), interval="30s", eager=True)[:N],
    'sensor_id': np.random.randint(1, 100, size=N),
    'temperature': np.random.normal(25.0, 5.0, size=N),
    'status': np.random.choice(['OK', 'WARN', 'CRITICAL'], size=N)
})

parquet_path = "sensor_data.parquet"
df.write_parquet(parquet_path, compression="zstd")

# 2. Execute SQL Analytical Query with DuckDB without loading full file into RAM
conn = duckdb.connect()
query = """
    SELECT 
        sensor_id, 
        AVG(temperature) as avg_temp,
        COUNT(CASE WHEN status = 'CRITICAL' THEN 1 END) as critical_count
    FROM 'sensor_data.parquet'
    WHERE temperature > 30.0
    GROUP BY sensor_id
    HAVING critical_count > 5
    ORDER BY critical_count DESC
    LIMIT 10
"""
result = conn.execute(query).pl()
print("DuckDB Parquet Query Result:")
print(result)
```

---

## 6. Production FastAPI Microservice Endpoint

Building a production API endpoint for streaming Parquet data ingestion and inference:

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import numpy as np

app = FastAPI(title="ML Pipeline Ingestion & Prediction API", version="2.0")

class PredictionRequest(BaseModel):
    feature_vector: list[float] = Field(..., min_length=3, max_length=3)
    user_id: int = Field(..., gt=0)

class PredictionResponse(BaseModel):
    user_id: int
    score: float
    status: str

@app.post("/api/v1/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    if any(np.isnan(request.feature_vector)):
        raise HTTPException(status_code=400, detail="Feature vector contains NaN values.")
        
    # Simulated Model Inference
    weights = np.array([0.5, -0.2, 0.8])
    score = float(np.dot(request.feature_vector, weights))
    status = "HIGH_RISK" if score > 1.0 else "LOW_RISK"
    
    return PredictionResponse(user_id=request.user_id, score=score, status=status)
```

---

## 7. Under the Hood

- **Parquet File Structure**: File Footer contains metadata (schema, column statistics `min`/`max` per row group). A reader parses the footer *first*, enabling row-group filtering prior to scanning data blocks.
- **FastAPI / Pydantic**: Pydantic v2 compiles data validation schemas into compiled Rust binaries, processing HTTP JSON payloads 5x faster than pure Python validators.

---

## 8. Engineering Perspective

- **Storage Cost Reduction**: Converting 100 GB CSV logs to ZSTD-compressed Parquet routinely reduces storage footprint to 8–15 GB (85%+ cost saving on S3).
- **SQL Optimization**: Use Window Functions (`ROW_NUMBER() OVER(PARTITION BY ... ORDER BY ...)`) instead of multiple self-joins for temporal sequence extraction.

---

## 9. Common Mistakes

1. **Storing ML Datasets as CSV or Pickle**: Pickle files are susceptible to arbitrary code execution vulnerabilities and Python-version breakage. CSV loses type definitions and is slow. Use Parquet or Feather.
2. **N+1 SQL Queries**: Executing single SQL queries inside a loop for each feature row instead of vectorized batch SQL queries.

---

## 10. Interview Questions

### Q1: Compare Parquet vs ORC vs Feather (Arrow IPC) file formats.
**Answer**: Parquet is optimized for heavy write-once, read-many analytical queries (Spark, DuckDB) with heavy compression. ORC is popular in the Hadoop/Hive ecosystem. Feather (Arrow IPC) is a uncompressed/lightly-compressed memory-mapped format optimized for ultra-fast inter-process communication between Python/R/Rust without serialization overhead.

---

## 11. Exercises

1. **SQL**: Write a SQL query using DuckDB window functions to compute 7-day exponential moving averages for stock prices.
2. **API**: Build a FastAPI endpoint that accepts multi-part form file uploads of Parquet files, parses them in Polars, and returns model prediction summaries.

---

## 12. Mini Project: High-Performance Data Ingestion Engine

Write a Python script `ingest_engine.py` that reads streaming JSON payload logs, validates them via Pydantic, batches them into 100k-row chunks, and appends them to a partitioned Parquet dataset on disk.

---

## 13. Capstone Integration

Powers the data loading utilities across `api/main.py` and `src/document_classification/pipeline.py`.
