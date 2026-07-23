# 🕸️ Chapter 15: Graph Neural Networks (GNNs) & Network Analytics

## 15.1 Graph Mathematics & Representations

Many real-world relational domain problems (fraud rings, social networks, molecular structures, financial transaction networks) are represented naturally as non-Euclidean graphs $\mathcal{G} = (\mathcal{V}, \mathcal{E})$.

```
                              Graph Topology G = (V, E)
                                      (Node 1)
                                      ╱      ╲
                                    ╱          ╲
                             (Node 2) ───────── (Node 3)
                                │                  │
                             (Node 4) ───────── (Node 5)
```

### Graph Matrices:
1. **Node Feature Matrix** $X \in \mathbb{R}^{N \times d}$: $N$ nodes with $d$ feature attributes per node.
2. **Adjacency Matrix** $A \in \mathbb{R}^{N \times N}$: Binary matrix indicating edges:
   $$A_{i, j} = \begin{cases} 1 & \text{if } (v_i, v_j) \in \mathcal{E} \\ 0 & \text{otherwise} \end{cases}$$
3. **Degree Matrix** $D \in \mathbb{R}^{N \times N}$: Diagonal matrix recording node edge counts $D_{i, i} = \sum_j A_{i, j}$.
4. **Graph Laplacian Matrix** $L \in \mathbb{R}^{N \times N}$:
   $$L = D - A, \quad \text{Symmetric Normalized Laplacian } L^{\text{sys}} = I - D^{-1/2} A D^{-1/2}$$

---

## 15.2 Message Passing Framework

Traditional neural networks process node features $x_i$ independently, ignoring graph topology. **Graph Neural Networks (GNNs)** use **message passing** where each node aggregates feature embeddings from its immediate neighborhood $\mathcal{N}(v)$.

```
                           GNN Message Passing (Layer l to l+1)
                                    Neighborhood Aggregate
                    Node 2 ──┐
                    Node 3 ──┼──► Aggregate M_v ──► Update Function ──► Node v's New Embedding
                    Node 4 ──┘                             (W, σ)              h_v^(l+1)
```

### General Message Passing Formula:
$$m_v^{(l+1)} = \text{AGGREGATE}^{(l+1)} \left( \left\{ h_u^{(l)} \mid u \in \mathcal{N}(v) \right\} \right)$$
$$h_v^{(l+1)} = \text{UPDATE}^{(l+1)} \left( h_v^{(l)}, m_v^{(l+1)} \right)$$

---

## 15.3 Graph Convolutional Networks (GCN)

Kipf & Welling (2017) formulated **Graph Convolutional Networks (GCN)** by applying a 1st-order localized spectral graph convolution approximation.

### GCN Layer Update Equation:
$$H^{(l+1)} = \sigma \left( \tilde{D}^{-1/2} \tilde{A} \tilde{D}^{-1/2} H^{(l)} W^{(l)} \right)$$

- $\tilde{A} = A + I_N$: Adjacency matrix with added self-loops (ensures node includes its own current feature during aggregation).
- $\tilde{D}_{i, i} = \sum_j \tilde{A}_{i, j}$: Degree matrix of $\tilde{A}$.
- $\tilde{D}^{-1/2} \tilde{A} \tilde{D}^{-1/2}$: Symmetric normalization preventing high-degree hub nodes from exploding gradient magnitudes.
- $W^{(l)}$: Trainable weight transformation matrix.
- $\sigma$: Non-linear activation function (e.g. ReLU).

---

## 15.4 Graph Attention Networks (GAT)

Standard GCN treats all neighbor connections equally using fixed degree normalization. **Graph Attention Networks (GAT)** compute dynamic attention weights $\alpha_{ij}$ over neighbor edges:

$$h_i^{(l+1)} = \sigma \left( \sum_{j \in \mathcal{N}(i)} \alpha_{ij} W^{(l)} h_j^{(l)} \right)$$

```
                                GAT Edge Attention Weights
                                         Node j
                                        ╱
                        Attention α_ij ╱ 
                                      ▼
                                   Node i
```

### Scaled Attention Coefficients:
$$\alpha_{ij} = \frac{\exp\left( \text{LeakyReLU}\left( \mathbf{a}^T [W h_i \,\|\, W h_j] \right) \right)}{\sum_{k \in \mathcal{N}(i)} \exp\left( \text{LeakyReLU}\left( \mathbf{a}^T [W h_i \,\|\, W h_k] \right) \right)}$$

where $\mathbf{a}$ is a learnable attention parameter vector, and $\|$ represents vector concatenation.

---

## 15.5 Enterprise Applications in Fraud & Finance

1. **Fraud Ring & Synthetic Identity Detection**: Individual credit card transactions might appear normal, but a graph topology connecting shared IP addresses, phone numbers, and bank accounts reveals organized fraud rings.
2. **Anti-Money Laundering (AML)**: Traces complex multi-hop layering transfers across bank account nodes using GNN node classification.

```python
# PyTorch Geometric GCN Example
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class FraudGCN(torch.nn.Module):
    def __init__(self, in_feats, hidden_dim, num_classes):
        super().__init__()
        self.conv1 = GCNConv(in_feats, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, num_classes)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)
```

---

## ⚓ Repository Code Reference
- See [`src/fraud_detection/pipeline.py`](file:///e:/Downloads/ML_only/src/fraud_detection/pipeline.py) for network velocity features.
