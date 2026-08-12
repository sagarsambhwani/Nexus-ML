# Chapter 13: Advanced Visualization Techniques for EDA

---

## 1. Big Picture

Basic plots (histograms, scatter plots) give you a first glance, but complex datasets often hide patterns that only **rich, multi‑dimensional visualizations** can reveal. Advanced visualizations let engineers spot interactions, outliers, and distribution quirks that are invisible in flat tables.

## 2. Intuition

Think of a visualization as a *lens* that can be tuned: rotate, zoom, color‑code, or animate. By mapping additional dimensions to hue, size, or animation frames, you expose hidden structure without writing extra code.

## 3. Core Techniques

| Technique | When to Use | Key Libraries |
|-----------|-------------|----------------|
| **Pair‑Plot Matrix** (corner‑grid) | Moderate‑size tabular data (≤ 30 columns) | `seaborn.pairplot`, `pandas‑profiling` |
| **Parallel Coordinates** | High‑dimensional numeric data | `plotly.express.parallel_coordinates`, `mpl‑toolkits` |
| **Interactive Heatmaps** (zoomable) | Correlation or similarity matrices | `plotly.graph_objects.Heatmap`, `holoviews` |
| **Violin / Boxen Plots** | Distribution comparison across categorical groups | `seaborn.violinplot`, `plotly.express.box` |
| **3‑D Scatter / Surface** | When you have 2‑3 numeric features and want to see spatial relationships | `plotly.express.scatter_3d`, `pyvista` |
| **Geo‑Spatial Maps** | Location‑aware data (e.g., sales by region) | `geopandas`, `folium`, `kepler.gl` |

## 4. Code Sketch (Plotly Parallel Coordinates)
```python
import pandas as pd
import plotly.express as px

# Assume df is a numeric DataFrame with ~10 columns
fig = px.parallel_coordinates(
    df,
    dimensions=[col for col in df.columns if df[col].dtype != "object"],
    color="target",  # optional: color by label
    color_continuous_scale=px.colors.sequential.Viridis,
)
fig.show()
```

## 5. Practical Tips
- **Pre‑process** numeric columns (scale to 0‑1) to improve visual contrast.
- **Filter** heavy‑weight dimensions (e.g., drop low‑variance columns) to avoid clutter.
- **Link** selections on one plot to filter data on others (brush‑and‑link).
- **Export** interactive HTML files for sharing with non‑technical stakeholders.

## 6. Real‑World Example
A fraud‑detection team visualized transaction amounts vs. time‑of‑day using a 3‑D scatter with color encoding for merchant category. The plot revealed a tight cluster of high‑value purchases at 02:00 AM for a specific category—later identified as a bot‑driven attack.

---

**Key Takeaway:** Leverage multi‑dimensional, interactive visualizations to turn raw numbers into intuitive insights that guide downstream feature engineering and modeling.
