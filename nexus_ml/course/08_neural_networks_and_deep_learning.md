# 🧠 Chapter 8: Neural Networks & Deep Learning Foundations

## 8.1 The Artificial Neuron (Perceptron)
An Artificial Neuron (Perceptron) is the fundamental building block of deep neural networks. It computes a weighted linear combination of input features followed by a non-linear activation function:

$$z = \sum_{i=1}^d w_i x_i + b = \mathbf{w}^T \mathbf{x} + b$$
$$a = \sigma(z)$$

```
Inputs (x1, x2, x3) ──► Weighted Sum (z = w^T x + b) ──► Activation Function (a = σ(z)) ──► Output (a)
```

---

## 8.2 Activation Functions

Non-linear activation functions enable neural networks to approximate arbitrary non-linear functions (Universal Approximation Theorem).

```
        Sigmoid                         ReLU                           GELU
     1 ┼       ╭───                  ┼      ╱                       ┼       ╭───
       │     ╭─╯                     │     ╱                        │     ╭─╯
   0.5 ┼────┼──────             0.0  ┼────╱                    0.0  ┼───╱───────
       │  ╭─╯                        │   ╱                          │ ╭─╯
     0 ┼──╯                          ┼──╱                           ┼─╯
```

### 1. Sigmoid Function
$$\sigma(z) = \frac{1}{1 + e^{-z}}, \quad \sigma'(z) = \sigma(z)(1 - \sigma(z))$$
- **Use Case**: Binary classification output layer ($a \in [0, 1]$).
- **Drawback**: Suffer from **vanishing gradients** when $|z|$ is large.

### 2. Rectified Linear Unit (ReLU)
$$\text{ReLU}(z) = \max(0, z), \quad \text{ReLU}'(z) = \begin{cases} 1 & \text{if } z > 0 \\ 0 & \text{if } z \le 0 \end{cases}$$
- **Use Case**: Default activation function for hidden layers in deep networks.
- **Drawback**: "Dying ReLU" problem when gradients become zero for negative inputs.

### 3. Gaussian Error Linear Unit (GELU) & Swish
$$\text{GELU}(z) = z \cdot \Phi(z) = z \cdot P(X \le z), \quad X \sim N(0, 1)$$
$$\text{Swish}(z) = z \cdot \sigma(\beta z)$$
- **Use Case**: Modern standard in Transformer models (BERT, GPT, RoBERTa).

---

## 8.3 Backpropagation Mathematics

Backpropagation uses the multivariate calculus **chain rule** to compute gradients of the loss function $\mathcal{L}$ with respect to network weights $W^{(l)}$ and biases $b^{(l)}$.

### Forward Pass:
$$z^{(l)} = W^{(l)} a^{(l-1)} + b^{(l)}, \quad a^{(l)} = \sigma(z^{(l)})$$

### Backward Pass (Gradient Chain Rule):
Define error delta at layer $l$:
$$\delta^{(l)} = \frac{\partial \mathcal{L}}{\partial z^{(l)}} = \delta^{(l+1)} (W^{(l+1)})^T \odot \sigma'(z^{(l)})$$

Gradients with respect to parameters:
$$\frac{\partial \mathcal{L}}{\partial W^{(l)}} = (a^{(l-1)})^T \delta^{(l)}, \quad \frac{\partial \mathcal{L}}{\partial b^{(l)}} = \sum_{i=1}^N \delta_i^{(l)}$$

---

## 8.4 Modern Optimizers

Gradient descent updates weights in the direction of steepest descent:
$$W^{(t+1)} = W^{(t)} - \eta \nabla_W \mathcal{L}$$

### 1. Stochastic Gradient Descent with Momentum (SGD+Momentum)
Accumulates past gradients to accelerate through flat regions and damp velocity oscillations:
$$v_{t} = \beta v_{t-1} + (1 - \beta) g_t, \quad W^{(t+1)} = W^{(t)} - \eta v_t$$

### 2. Adam (Adaptive Moment Estimation)
Combines 1st-moment (mean gradient) and 2nd-moment (uncentered variance gradient) exponential moving averages:
$$m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t, \quad v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2$$
$$\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}$$
$$W^{(t+1)} = W^{(t)} - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t$$

### 3. AdamW (Decoupled Weight Decay)
Standard Adam applies $L_2$ regularization by adding weight decay directly to gradients $g_t$. **AdamW** decouples weight decay from gradient moment updates, yielding vastly superior generalization in Transformers and Deep CNNs:
$$W^{(t+1)} = W^{(t)} - \eta \left( \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} + \lambda W^{(t)} \right)$$

---

## 8.5 Deep Learning Architectures

### 1. Convolutional Neural Networks (CNNs) & ResNet
CNNs use spatial convolution kernels to extract local visual patterns (edges, textures, shapes). **ResNet** introduces residual skip connections:
$$y = \mathcal{F}(x, \{W_i\}) + x$$
Skip connections allow gradients to flow directly back through deep networks, enabling models with 100+ layers without vanishing gradients.

### 2. Transformers & Scaled Dot-Product Attention
Transformers replace recurrent connections with parallelizable **Self-Attention**:

$$\text{Attention}(Q, K, V) = \text{Softmax}\left( \frac{Q K^T}{\sqrt{d_k}} \right) V$$

where Query ($Q$), Key ($K$), and Value ($V$) linear projections capture global token dependencies across sequences.

---

## ⚓ Repository Code Reference
- See [`src/defect_detection/pipeline.py`](file:///e:/Downloads/ML_only/src/defect_detection/pipeline.py) for computer vision feature extraction principles.
- See [`src/sentiment_analysis/pipeline.py`](file:///e:/Downloads/ML_only/src/sentiment_analysis/pipeline.py) for text feature classification architectures.
