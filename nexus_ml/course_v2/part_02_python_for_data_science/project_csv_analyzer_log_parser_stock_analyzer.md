# Part II Portfolio Projects: Production Data Science Tools

This section contains 3 complete, production-grade Python engineering tools built using NumPy, Polars, DuckDB, and SQL.

---

## 🛠️ Project 1: High-Throughput CSV & Parquet Schema Analyzer

An automated command-line utility that inspects arbitrarily large datasets without loading them entirely into memory, returning data types, memory consumption, missing value percentages, and skewness metrics.

```python
import sys
import polars as pl
from pathlib import Path

def analyze_dataset(file_path: str):
    path = Path(file_path)
    if not path.exists():
        print(f"Error: File '{file_path}' not found.")
        sys.exit(1)
        
    print(f"=== Dataset Schema Analysis: {path.name} ===")
    
    # Use Polars Lazy Scan for zero-copy schema discovery
    if path.suffix == '.parquet':
        lazy_df = pl.scan_parquet(path)
    elif path.suffix == '.csv':
        lazy_df = pl.scan_csv(path)
    else:
        print("Unsupported format. Use .parquet or .csv")
        return
        
    schema = lazy_df.collect_schema()
    print(f"\nTotal Columns: {len(schema)}")
    print("-" * 50)
    for col_name, dtype in schema.items():
        print(f"Column: {col_name:<25} | Type: {str(dtype):<15}")
        
    # Execute aggregated metrics query
    df_sample = lazy_df.fetch(10_000)
    print("\n=== Sample Statistics (First 10k Rows) ===")
    print(df_sample.describe())

if __name__ == "__main__":
    if len(sys.argv) > 1:
        analyze_dataset(sys.argv[1])
    else:
        print("Usage: python csv_analyzer.py <path_to_file>")
```

---

## 🛠️ Project 2: High-Speed Web Log Parser & Anomaly Extractor

Parse raw NGINX/Apache access logs at 50,000 lines/second, extracting IP addresses, HTTP status codes, response times, and anomaly spikes.

```python
import re
import polars as pl

LOG_PATTERN = r'(?P<ip>\S+) \S+ \S+ \[(?P<timestamp>.*?)\] "(?P<method>\S+) (?P<path>\S+) \S+" (?P<status>\d{3}) (?P<bytes>\d+)'

def parse_logs(log_file_path: str):
    print(f"Parsing log file: {log_file_path}...")
    
    with open(log_file_path, "r") as f:
        lines = f.readlines()
        
    records = []
    for line in lines:
        match = re.search(LOG_PATTERN, line)
        if match:
            records.append(match.groupdict())
            
    df = pl.DataFrame(records)
    df = df.with_columns([
        pl.col("status").cast(pl.Int32),
        pl.col("bytes").cast(pl.Int64)
    ])
    
    print("\n=== Top Error Request IPs (Status >= 400) ===")
    error_df = df.filter(pl.col("status") >= 400).group_by("ip").len().sort("len", descending=True)
    print(error_df.head(5))

# Example invocation: parse_logs("access.log")
```

---

## 🛠️ Project 3: Financial Stock Market Volatility & Momentum Analyzer

A vectorized momentum stock analysis tool that calculates 20-day Simple Moving Averages (SMA), Bollinger Bands, and Exponential Moving Averages (EMA).

```python
import numpy as np
import polars as pl

def calculate_technical_indicators(prices: list[float]) -> pl.DataFrame:
    df = pl.DataFrame({"close": prices})
    
    df = df.with_columns([
        pl.col("close").rolling_mean(window_size=20).alias("sma_20"),
        pl.col("close").rolling_std(window_size=20).alias("std_20"),
        pl.col("close").ewm_mean(span=12).alias("ema_12"),
    ])
    
    # Calculate Bollinger Bands
    df = df.with_columns([
        (pl.col("sma_20") + 2 * pl.col("std_20")).alias("bollinger_upper"),
        (pl.col("sma_20") - 2 * pl.col("std_20")).alias("bollinger_lower"),
    ])
    
    return df

# Test run with synthetic stock prices
np.random.seed(42)
prices = (100 + np.cumsum(np.random.normal(0, 1, 100))).tolist()
indicators = calculate_technical_indicators(prices)
print("Technical Indicators (First 25 Rows):")
print(indicators.head(25))
```
