# 🌐 Chapter 28: Geospatial ML & Spatiotemporal Modeling

## 28.1 Spatial Autocorrelation & Tobler's First Law
Tobler's First Law of Geography states: *"Everything is related to everything else, but near things are more related than distant things."*

Spatial data exhibits **spatial autocorrelation**, violating the i.i.d. (independent and identically distributed) assumption of standard ML models.

```
       Positive Spatial Autocorrelation             Negative Spatial Autocorrelation
       (Clustered Homogeneous Clusters)            (Checkerboard Spatial Disparity)
            ■ ■ ■ ■   □ □ □ □                           ■ □ ■ □   □ ■ □ ■
            ■ ■ ■ ■   □ □ □ □                           □ ■ □ ■   ■ □ ■ □
```

### Moran's $I$ Statistic (Spatial Autocorrelation):
$$I = \frac{N}{S_0} \frac{\sum_{i=1}^N \sum_{j=1}^N w_{ij} (x_i - \bar{x})(x_j - \bar{x})}{\sum_{i=1}^N (x_i - \bar{x})^2}$$

- $I > 0$: Positive spatial clustering.
- $I < 0$: Dispersion / Checkerboard pattern.
- $I \approx 0$: Random spatial distribution.

---

## 28.2 Geospatial Indexing: Uber H3 & S2

Geospatial coordinates (Latitude, Longitude) must be discretized into spatial index cells for fast aggregation and spatial join operations.

```
                             Uber H3 Hexagonal Grid
                                      ╭───╮
                                  ╭───┤ H ├───╮
                                  │ H ├───┤ H │
                                  ├───┤ H ├───┤
                                  │ H ├───┤ H │
                                  ╰───┤ H ├───╯
                                      ╰───╮
```

### Uber H3 (Hexagonal Hierarchical Spatial Index):
- **Why Hexagons?**: Unlike squares (where diagonal neighbors are farther than edge neighbors), hexagons have **equal centroid distances to all 6 adjacent neighbors**, simplifying spatial smoothing and distance calculations.

---

## 28.3 Spatiotemporal Deep Learning: ConvLSTM

ConvLSTM (Shi et al. 2015) extends recurrent LSTM cells by replacing internal matrix multiplications with 2D spatial convolutions:

$$\mathcal{X}_t, \mathcal{H}_t, \mathcal{C}_t \in \mathbb{R}^{C \times H \times W}$$
$$f_t = \sigma\left( W_{xf} * \mathcal{X}_t + W_{hf} * \mathcal{H}_{t-1} + W_{cf} \odot \mathcal{C}_{t-1} + b_f \right)$$
$$i_t = \sigma\left( W_{xi} * \mathcal{X}_t + W_{hi} * \mathcal{H}_{t-1} + W_{ci} \odot \mathcal{C}_{t-1} + b_i \right)$$
$$\mathcal{C}_t = f_t \odot \mathcal{C}_{t-1} + i_t \odot \tanh\left( W_{xc} * \mathcal{X}_t + W_{hc} * \mathcal{H}_{t-1} + b_c \right)$$

where $*$ denotes 2D spatial convolution operator.

---

## ⚓ Repository Code Reference
- See [`src/house_prices/pipeline.py`](file:///e:/Downloads/ML_only/src/house_prices/pipeline.py) for spatial distance and location score features (`dist_city_km`).
