"""Complete Convolutional Neural Network (CNN) and Backpropagation from Scratch in Pure NumPy.

Implements exact analytical forward and backward gradient passes for:
1. Conv2D (Arbitrary channels, kernels, padding, and stride)
2. ReLU Activation
3. MaxPool2D (Argmax gradient routing)
4. Flatten Layer
5. Dense (Fully Connected) Linear Layer
6. Softmax with Cross-Entropy Loss
"""

import numpy as np
from typing import Tuple, List, Dict

class Conv2DLayer:
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        # He / Kaiming Normal initialization
        scale = np.sqrt(2.0 / (in_channels * kernel_size * kernel_size))
        self.W = np.random.randn(out_channels, in_channels, kernel_size, kernel_size).astype(np.float32) * scale
        self.b = np.zeros(out_channels, dtype=np.float32)
        
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)
        self.x = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass for 2D convolution."""
        self.x = x
        N, C, H, W = x.shape
        k = self.kernel_size
        
        if self.padding > 0:
            padded_x = np.pad(x, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)), mode='constant')
        else:
            padded_x = x
            
        out_H = (H + 2 * self.padding - k) // self.stride + 1
        out_W = (W + 2 * self.padding - k) // self.stride + 1
        out = np.zeros((N, self.out_channels, out_H, out_W), dtype=np.float32)
        
        for n in range(N):
            for c_out in range(self.out_channels):
                for i in range(out_H):
                    for j in range(out_W):
                        h_s = i * self.stride
                        w_s = j * self.stride
                        patch = padded_x[n, :, h_s:h_s+k, w_s:w_s+k]
                        out[n, c_out, i, j] = np.sum(patch * self.W[c_out]) + self.b[c_out]
                        
        return out

    def backward(self, dout: np.ndarray) -> np.ndarray:
        """Backward analytical gradient pass for Conv2D."""
        N, C, H, W = self.x.shape
        k = self.kernel_size
        out_H, out_W = dout.shape[2], dout.shape[3]
        
        self.dW.fill(0)
        self.db.fill(0)
        
        if self.padding > 0:
            padded_x = np.pad(self.x, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)), mode='constant')
            dx_padded = np.zeros_like(padded_x)
        else:
            padded_x = self.x
            dx_padded = np.zeros_like(self.x)
            
        # Compute db
        self.db = np.sum(dout, axis=(0, 2, 3))
        
        # Compute dW and dx
        for n in range(N):
            for c_out in range(self.out_channels):
                for i in range(out_H):
                    for j in range(out_W):
                        h_s = i * self.stride
                        w_s = j * self.stride
                        
                        patch = padded_x[n, :, h_s:h_s+k, w_s:w_s+k]
                        self.dW[c_out] += patch * dout[n, c_out, i, j]
                        dx_padded[n, :, h_s:h_s+k, w_s:w_s+k] += self.W[c_out] * dout[n, c_out, i, j]
                        
        if self.padding > 0:
            dx = dx_padded[:, :, self.padding:-self.padding, self.padding:-self.padding]
        else:
            dx = dx_padded
            
        return dx

class ReLULayer:
    def __init__(self):
        self.x = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.x = x
        return np.maximum(0, x)

    def backward(self, dout: np.ndarray) -> np.ndarray:
        return dout * (self.x > 0)

class MaxPool2DLayer:
    def __init__(self, pool_size: int = 2, stride: int = 2):
        self.pool_size = pool_size
        self.stride = stride
        self.x = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.x = x
        N, C, H, W = x.shape
        out_H = (H - self.pool_size) // self.stride + 1
        out_W = (W - self.pool_size) // self.stride + 1
        out = np.zeros((N, C, out_H, out_W), dtype=np.float32)
        
        for n in range(N):
            for c in range(C):
                for i in range(out_H):
                    for j in range(out_W):
                        h_s = i * self.stride
                        w_s = j * self.stride
                        patch = x[n, c, h_s:h_s+self.pool_size, w_s:w_s+self.pool_size]
                        out[n, c, i, j] = np.max(patch)
        return out

    def backward(self, dout: np.ndarray) -> np.ndarray:
        N, C, H, W = self.x.shape
        dx = np.zeros_like(self.x)
        out_H, out_W = dout.shape[2], dout.shape[3]
        
        for n in range(N):
            for c in range(C):
                for i in range(out_H):
                    for j in range(out_W):
                        h_s = i * self.stride
                        w_s = j * self.stride
                        patch = self.x[n, c, h_s:h_s+self.pool_size, w_s:w_s+self.pool_size]
                        max_idx = np.unravel_index(np.argmax(patch), patch.shape)
                        dx[n, c, h_s + max_idx[0], w_s + max_idx[1]] += dout[n, c, i, j]
        return dx

class DenseLayer:
    def __init__(self, in_features: int, out_features: int):
        scale = np.sqrt(2.0 / in_features)
        self.W = np.random.randn(in_features, out_features).astype(np.float32) * scale
        self.b = np.zeros(out_features, dtype=np.float32)
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)
        self.x = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.x = x.reshape(x.shape[0], -1)
        return np.dot(self.x, self.W) + self.b

    def backward(self, dout: np.ndarray) -> np.ndarray:
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)
        dx = np.dot(dout, self.W.T)
        return dx.reshape(self.x.shape)

def softmax_cross_entropy_loss(logits: np.ndarray, labels: np.ndarray) -> Tuple[float, np.ndarray]:
    """Computes stable Softmax Cross-Entropy loss and analytical gradient."""
    N = logits.shape[0]
    # Numerical stability shift
    shifted_logits = logits - np.max(logits, axis=1, keepdims=True)
    exp_scores = np.exp(shifted_logits)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    
    loss = -np.sum(np.log(probs[np.arange(N), labels] + 1e-12)) / N
    dlogits = probs.copy()
    dlogits[np.arange(N), labels] -= 1.0
    dlogits /= N
    return loss, dlogits

class ScratchCNN:
    def __init__(self, in_channels: int = 1, num_classes: int = 2):
        self.conv1 = Conv2DLayer(in_channels=in_channels, out_channels=4, kernel_size=3, padding=1)
        self.relu1 = ReLULayer()
        self.pool1 = MaxPool2DLayer(pool_size=2, stride=2)
        self.fc = DenseLayer(in_features=4 * 4 * 4, out_features=num_classes)

    def forward(self, x: np.ndarray) -> np.ndarray:
        c1 = self.conv1.forward(x)
        r1 = self.relu1.forward(c1)
        p1 = self.pool1.forward(r1)
        self.p1_shape = p1.shape
        logits = self.fc.forward(p1)
        return logits

    def backward(self, dlogits: np.ndarray):
        dfc = self.fc.backward(dlogits)
        dpool = self.pool1.backward(dfc.reshape(self.p1_shape))
        drelu = self.relu1.backward(dpool)
        self.conv1.backward(drelu)

    def step(self, lr: float = 0.01):
        """SGD weight update."""
        self.conv1.W -= lr * self.conv1.dW
        self.conv1.b -= lr * self.conv1.db
        self.fc.W -= lr * self.fc.dW
        self.fc.b -= lr * self.fc.db

if __name__ == "__main__":
    np.random.seed(42)
    # Generate 16 synthetic 8x8 images for binary classification
    X = np.random.randn(16, 1, 8, 8).astype(np.float32)
    y = np.random.choice([0, 1], size=16)
    
    model = ScratchCNN(in_channels=1, num_classes=2)
    
    print("=== Scratch Pure NumPy CNN Forward & Backward Pass ===")
    initial_loss = 0.0
    for epoch in range(10):
        logits = model.forward(X)
        loss, dlogits = softmax_cross_entropy_loss(logits, y)
        if epoch == 0:
            initial_loss = loss
        model.backward(dlogits)
        model.step(lr=0.05)
        
    print(f"Initial Loss: {initial_loss:.4f} ──► Final Loss after 10 epochs: {loss:.4f}")
    assert loss < initial_loss, "Model failed to minimize loss through backpropagation!"
    print("[OK] Pure NumPy CNN forward & backward pass verified successfully!")
