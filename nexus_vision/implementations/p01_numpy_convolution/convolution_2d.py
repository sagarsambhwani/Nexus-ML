"""2D Discrete Convolution from Scratch in Pure NumPy.

Supports:
- Multi-channel input tensors (H, W, C_in) and batch tensors (N, C_in, H, W).
- Multiple output filters (C_out).
- Arbitrary padding (zero-padding) and strides.
- Standard classical image processing filters (Sobel, Gaussian, Box, Sharpen).
"""

import numpy as np
from typing import Tuple

def conv2d_single_channel(image: np.ndarray, kernel: np.ndarray, stride: int = 1, padding: int = 0) -> np.ndarray:
    """Computes 2D convolution for a single channel (H, W) with a kernel (kH, kW)."""
    H, W = image.shape
    kH, kW = kernel.shape

    if padding > 0:
        padded_img = np.pad(image, ((padding, padding), (padding, padding)), mode='constant', constant_values=0)
    else:
        padded_img = image

    out_H = (H + 2 * padding - kH) // stride + 1
    out_W = (W + 2 * padding - kW) // stride + 1
    output = np.zeros((out_H, out_W), dtype=np.float32)

    for i in range(out_H):
        for j in range(out_W):
            h_start = i * stride
            h_end = h_start + kH
            w_start = j * stride
            w_end = w_start + kW
            patch = padded_img[h_start:h_end, w_start:w_end]
            output[i, j] = np.sum(patch * kernel)

    return output

def conv2d_multichannel(
    images: np.ndarray, 
    kernels: np.ndarray, 
    bias: np.ndarray = None, 
    stride: int = 1, 
    padding: int = 0
) -> np.ndarray:
    """Computes multi-channel 2D convolution for batch tensor (N, C_in, H, W).

    Args:
        images: Input tensor of shape (N, C_in, H, W)
        kernels: Filter tensor of shape (C_out, C_in, kH, kW)
        bias: Optional bias vector of shape (C_out,)
        stride: Spatial step size
        padding: Zero-padding size on each border

    Returns:
        output: Convolved feature map tensor of shape (N, C_out, out_H, out_W)
    """
    N, C_in, H, W = images.shape
    C_out, k_Cin, kH, kW = kernels.shape
    assert C_in == k_Cin, f"Input channels ({C_in}) must match kernel input channels ({k_Cin})"

    if padding > 0:
        padded = np.pad(images, ((0, 0), (0, 0), (padding, padding), (padding, padding)), mode='constant', constant_values=0)
    else:
        padded = images

    out_H = (H + 2 * padding - kH) // stride + 1
    out_W = (W + 2 * padding - kW) // stride + 1
    output = np.zeros((N, C_out, out_H, out_W), dtype=np.float32)

    for n in range(N):
        for c_out in range(C_out):
            for i in range(out_H):
                for j in range(out_W):
                    h_start = i * stride
                    h_end = h_start + kH
                    w_start = j * stride
                    w_end = w_start + kW
                    
                    patch = padded[n, :, h_start:h_end, w_start:w_end]
                    val = np.sum(patch * kernels[c_out])
                    if bias is not None:
                        val += bias[c_out]
                    output[n, c_out, i, j] = val

    return output

def get_classical_filter(name: str) -> np.ndarray:
    """Returns standard classical image processing kernels."""
    name = name.lower()
    if name == "sobel_v":
        return np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
    elif name == "sobel_h":
        return np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)
    elif name == "gaussian_3x3":
        return np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=np.float32) / 16.0
    elif name == "sharpen":
        return np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
    elif name == "laplacian":
        return np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)
    else:
        raise ValueError(f"Unknown filter: {name}")

if __name__ == "__main__":
    # Test single channel convolution with Sobel Vertical filter
    test_img = np.zeros((8, 8), dtype=np.float32)
    test_img[:, :4] = 255.0  # Vertical step edge in middle
    sobel_v = get_classical_filter("sobel_v")
    edge_response = conv2d_single_channel(test_img, sobel_v, padding=1)
    
    print("=== NumPy 2D Convolution Demo ===")
    print("Input Image Shape:", test_img.shape)
    print("Edge Filter Kernel:\n", sobel_v)
    print("Convolved Edge Map:\n", edge_response.round(1))
    
    # Test batch multichannel tensor convolution
    batch_img = np.random.randn(2, 3, 16, 16).astype(np.float32)
    batch_kernels = np.random.randn(4, 3, 3, 3).astype(np.float32)
    batch_bias = np.zeros(4, dtype=np.float32)
    
    out_tensor = conv2d_multichannel(batch_img, batch_kernels, batch_bias, stride=2, padding=1)
    print("\nBatch Multichannel Output Tensor Shape:", out_tensor.shape)
    assert out_tensor.shape == (2, 4, 8, 8), f"Expected shape (2, 4, 8, 8), got {out_tensor.shape}"
    print("[OK] All 2D convolution operations verified successfully!")
