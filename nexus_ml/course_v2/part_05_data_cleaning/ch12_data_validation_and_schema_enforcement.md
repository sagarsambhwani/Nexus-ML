# Chapter 12: Data Validation, Consistency Checks, and Schema Enforcement

---

## 1. Big Picture

In production pipelines data must not only be clean—it must also **conform** to expected contracts. Data validation catches schema drift, type mismatches, and logical inconsistencies before they corrupt downstream models.

## 2. Intuition

Think of data validation like linting for source code: it enforces a set of rules that keep the dataset trustworthy. By catching violations early you avoid costly model retraining failures.

## 3. Core Concepts

- **Schema Definition** – JSON Schema, Pandera, Great Expectations.
- **Constraint Types** – type checks, range checks, uniqueness, nullability, custom business rules.
- **Validation Strategies** – batch vs. streaming, offline vs. online.

## 4. Code Sketch (Python)
```python
import pandas as pd
import pandera as pa

schema = pa.DataFrameSchema({
    "user_id": pa.Column(pa.Int, nullable=False, unique=True),
    "signup_ts": pa.Column(pa.DateTime, nullable=False),
    "age": pa.Column(pa.Int, checks=pa.Check.in_range(0, 120)),
    "country": pa.Column(pa.String, checks=pa.Check.isin(["US", "CA", "GB"]))
})

# Validate a DataFrame
validated_df = schema.validate(df)
```

## 5. Operational Tips
- Run validation as a **pre‑commit** hook in your data ingestion CI pipeline.
- Store validation reports in a log‑centralized system (e.g., ELK) for auditing.
- Use **data contracts** between teams to formalize expectations.

## 6. Real‑World Example
A fintech company enforced a schema on transaction records. When a new partner introduced a missing `currency` field, the validator flagged the issue, preventing a downstream model from ingesting corrupted data and saving $200k in potential loss.

---

**Key Takeaway:** Rigorous data validation is the gatekeeper that ensures a clean, reliable foundation for all downstream ML work.
