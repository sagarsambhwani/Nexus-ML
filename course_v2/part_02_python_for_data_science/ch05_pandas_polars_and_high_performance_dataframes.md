# Chapter 05: Pandas, Polars, and High-Performance DataFrames

---

## 1. Big Picture

Tabular data is the lifeblood of enterprise machine learning. While **Pandas** has historically been the standard, its eager execution model and single-threaded Python GIL constraints make it slow and memory-inefficient on large datasets ($> 10\text{ GB}$). Modern ML infrastructure uses **Polars**—a multithreaded, lazy evaluation DataFrame library built in Rust using **Apache Arrow** columnar memory.

This chapter covers high-performance DataFrame operations, vectorization, memory optimization, and Polars lazy query optimization.

---

## 2. Intuition

- **Pandas**: Eager execution. Like ordering a burger, eating it immediately, then ordering fries, eating them immediately. Every step allocates intermediate memory buffers on disk/RAM.
- **Polars**: Lazy evaluation. Like handing a full menu order to the kitchen chef. The chef optimizes the cooking workflow (query plan optimization), cooks everything in parallel, and delivers the finished dish in a single pass.

---

## 3. Visualization

```text
Pandas Eager Execution (Intermediate Copy Overhead):
  Raw Data ──► [ Filter ] ──► (Alloc Memory 1) ──► [ GroupBy ] ──► (Alloc Memory 2) ──► Result

Polars Lazy Execution (Logical Query Optimization):
  Raw Data ──┐
             ├──► [ Query Planner: Predicate Pushdown + Projection Pushdown ] ──► Optimized Multi-threaded SIMD Execution ──► Result
  Query Def ──┘
```

---

## 4. Mathematics

The memory required for a Pandas DataFrame with $N$ rows and $M$ columns of mixed data types is bounded below by:

$$\text{Memory}_{\text{Pandas}} \ge \sum_{j=1}^M N \cdot \text{sizeof}(\text{dtype}_j) + N \cdot M \cdot \text{PointerOverhead}$$

For Python `object` dtypes (strings), pointer overhead is 8 bytes per cell + 50+ bytes per string object on the Python heap.

In **Apache Arrow Columnar Format** (Polars), strings are stored in a single contiguous byte buffer with offset offsets:
$$\text{Memory}_{\text{Arrow}} = \sum_{j=1}^{M_{\text{numeric}}} N \cdot \text{sizeof}(\text{dtype}_j) + \text{TotalStringBytes} + (N + 1) \cdot 4\text{ bytes}$$

---

## 5. Python (Pandas vs Polars Benchmark)

Comparing filtering and aggregation performance:

```python
import pandas as pd
import polars as pl
import numpy as np
import time

N = 2_000_000

# Synthetic Data Generation
data = {
    'user_id': np.random.randint(1000, 9999, size=N),
    'category': np.random.choice(['Electronics', 'Clothing', 'Home', 'Books'], size=N),
    'amount': np.random.uniform(5.0, 500.0, size=N),
    'age': np.random.randint(18, 70, size=N)
}

# 1. Pandas Benchmark
df_pd = pd.DataFrame(data)
start = time.perf_counter()
res_pd = df_pd[(df_pd['age'] > 30) & (df_pd['category'] == 'Electronics')].groupby('user_id')['amount'].mean()
pandas_time = time.perf_counter() - start

# 2. Polars Lazy Benchmark
df_pl = pl.DataFrame(data)
start = time.perf_counter()
res_pl = (
    df_pl.lazy()
    .filter((pl.col('age') > 30) & (pl.col('category') == 'Electronics'))
    .group_by('user_id')
    .agg(pl.col('amount').mean())
    .collect()  # Triggers query optimization & execution
)
polars_time = time.perf_counter() - start

print(f"Pandas Time: {pandas_time:.4f}s | Polars Time: {polars_time:.4f}s")
print(f"Polars is {pandas_time / polars_time:.1f}x faster!")
```

---

## 6. Production Optimization Techniques

### Pandas Memory Reduction Strategy
```python
def optimize_pandas_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Downcasts numeric types and converts strings to categories."""
    for col in df.columns:
        col_type = df[col].dtype
        if col_type == 'int64':
            df[col] = pd.to_numeric(df[col], downcast='integer')
        elif col_type == 'float64':
            df[col] = pd.to_numeric(df[col], downcast='float')
        elif col_type == 'object':
            num_unique = df[col].nunique()
            if num_unique / len(df) < 0.5:
                df[col] = df[col].astype('category')
    return df
```

---

## 7. Under the Hood

- **Polars** uses Apache Arrow memory layout which alignment guarantees SIMD vector loading directly into CPU cache lines without pointer indirection.
- **Query Planner Optimizations**:
  - *Predicate Pushdown*: Filters rows at the file scan stage (e.g. Parquet row-group skipping) before loading columns into memory.
  - *Projection Pushdown*: Reads only the columns referenced in the query, ignoring unused columns.

---

## 8. Engineering Perspective

- **Throughput**: Polars scales linearly across all physical CPU cores without GIL bottlenecking.
- **Out-of-Core Processing**: Polars supports streaming execution (`.collect(streaming=True)`), processing datasets larger than system RAM.

---

## 9. Common Mistakes

1. **Chained Indexing in Pandas**: Using `df[df['a'] > 2]['b'] = 5` causing `SettingWithCopyWarning`. Use `df.loc[df['a'] > 2, 'b'] = 5`.
2. **Converting Polars to Pandas prematurely**: Negating all Rust/Arrow multithreaded performance benefits by calling `.to_pandas()` early in the pipeline.

---

## 10. Interview Questions

### Q1: Explain Predicate Pushdown and Projection Pushdown in query optimization.
**Answer**: Projection pushdown minimizes IO by reading only requested columns from storage into memory. Predicate pushdown pushes filtering conditions down to the storage scan layer so non-matching row groups are never read into memory at all.

---

## 11. Exercises

1. **Coding**: Implement a Polars lazy expression that computes rolling 7-day average sales and 30-day volatility per store location.
2. **Memory Profiling**: Compare RAM consumption between Pandas `category` dtype vs Polars `Enum` / `Categorical` dtype on a 5M row dataset.

---

## 12. Mini Project: Log Aggregator Service

Write a Polars streaming script `process_logs.py` that parses 50GB of web server log files in chunks, extracts HTTP status codes, and outputs hourly error rate metrics.

---

## 13. Capstone Integration

Integrated in `src/demand_forecasting/pipeline.py` for high-throughput temporal aggregations and rolling window feature extractions.
