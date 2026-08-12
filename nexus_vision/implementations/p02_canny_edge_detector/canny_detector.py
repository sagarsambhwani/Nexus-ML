"""5-Stage Canny Edge Detector Implementation from First Principles in Pure NumPy.

Stages:
1. Gaussian Smoothing (Noise Attenuation)
2. Gradient Magnitude & Direction (Sobel Operators)
3. Non-Maximum Suppression (NMS - Edge Thinning along gradient normal)
4. Double Thresholding (Categorizing Strong vs Weak vs Non-edge pixels)
5. Edge Tracking by Hysteresis (Connectivity Analysis)
"""

import numpy as np
from typing import Tuple

class CannyEdgeDetector:
    def __init__(self, low_threshold: float = 20.0, high_threshold: float = 50.0, sigma: float = 1.0):
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold
        self.sigma = sigma

    def gaussian_kernel(self, size: int = 5, sigma: float = 1.0) -> np.ndarray:
        """Generates a 2D Gaussian smoothing kernel."""
        k = size // 2
        y, x = np.mgrid[-k:k+1, -k:k+1]
        g = np.exp(-(x**2 + y**2) / (2 * sigma**2))
        return g / g.sum()

    def convolve(self, image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        """2D spatial convolution helper with reflection padding."""
        H, W = image.shape
        kH, kW = kernel.shape
        pad_h, pad_w = kH // 2, kW // 2
        
        padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='reflect')
        output = np.zeros((H, W), dtype=np.float32)
        
        for i in range(H):
            for j in range(W):
                patch = padded[i:i+kH, j:j+kW]
                output[i, j] = np.sum(patch * kernel)
        return output

    def compute_gradients(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Computes gradient magnitude and orientation using Sobel operators."""
        sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
        sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)
        
        Gx = self.convolve(image, sobel_x)
        Gy = self.convolve(image, sobel_y)
        
        magnitude = np.hypot(Gx, Gy)
        magnitude = (magnitude / (magnitude.max() + 1e-8)) * 255.0
        angle = np.arctan2(Gy, Gx) * (180.0 / np.pi)
        angle[angle < 0] += 180.0
        
        return magnitude, angle

    def non_maximum_suppression(self, magnitude: np.ndarray, angle: np.ndarray) -> np.ndarray:
        """Thins edges by suppressing non-peak pixels along gradient direction."""
        H, W = magnitude.shape
        suppressed = np.zeros((H, W), dtype=np.float32)
        
        for i in range(1, H - 1):
            for j in range(1, W - 1):
                ang = angle[i, j]
                val = magnitude[i, j]
                
                # Direction 0: Horizontal (0 deg)
                if (0 <= ang < 22.5) or (157.5 <= ang <= 180):
                    q = magnitude[i, j + 1]
                    r = magnitude[i, j - 1]
                # Direction 1: Diagonal / (45 deg)
                elif 22.5 <= ang < 67.5:
                    q = magnitude[i + 1, j - 1]
                    r = magnitude[i - 1, j + 1]
                # Direction 2: Vertical (90 deg)
                elif 67.5 <= ang < 112.5:
                    q = magnitude[i + 1, j]
                    r = magnitude[i - 1, j]
                # Direction 3: Diagonal \ (135 deg)
                elif 112.5 <= ang < 157.5:
                    q = magnitude[i - 1, j - 1]
                    r = magnitude[i + 1, j + 1]
                else:
                    q, r = 255, 255
                    
                if val >= q and val >= r:
                    suppressed[i, j] = val
                else:
                    suppressed[i, j] = 0.0
                    
        return suppressed

    def double_threshold(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Categorizes pixels into Strong (255), Weak (75), and Non-edge (0)."""
        strong = (image >= self.high_threshold)
        weak = (image >= self.low_threshold) & (image < self.high_threshold)
        
        res = np.zeros_like(image, dtype=np.uint8)
        res[strong] = 255
        res[weak] = 75
        return res, strong, weak

    def hysteresis_tracking(self, image: np.ndarray) -> np.ndarray:
        """Preserves weak edge pixels connected to strong edges."""
        H, W = image.shape
        output = np.copy(image)
        
        for i in range(1, H - 1):
            for j in range(1, W - 1):
                if output[i, j] == 75:  # Weak pixel
                    # Check 8-neighborhood for strong pixel
                    neighbors = output[i-1:i+2, j-1:j+2]
                    if (neighbors == 255).any():
                        output[i, j] = 255
                    else:
                        output[i, j] = 0
                        
        output[output != 255] = 0
        return output

    def detect(self, image: np.ndarray) -> np.ndarray:
        """Executes full 5-stage Canny Edge Detection pipeline."""
        # 1. Gaussian Blur
        kernel = self.gaussian_kernel(size=5, sigma=self.sigma)
        blurred = self.convolve(image, kernel)
        
        # 2. Gradient Magnitude & Orientation
        magnitude, angle = self.compute_gradients(blurred)
        
        # 3. Non-Maximum Suppression
        nms = self.non_maximum_suppression(magnitude, angle)
        
        # 4. Double Thresholding
        thresh, _, _ = self.double_threshold(nms)
        
        # 5. Hysteresis Tracking
        edges = self.hysteresis_tracking(thresh)
        return edges

if __name__ == "__main__":
    # Create synthetic geometric shapes (rectangle & circle)
    test_img = np.zeros((32, 32), dtype=np.float32)
    test_img[8:24, 8:24] = 200.0  # Rectangle
    
    detector = CannyEdgeDetector(low_threshold=30, high_threshold=80, sigma=1.0)
    edges = detector.detect(test_img)
    
    print("=== NumPy Canny Edge Detector ===")
    print("Input Test Image Shape:", test_img.shape)
    print("Detected Binary Edge Map (Sample 16x16 window):\n", (edges[6:22, 6:22] > 0).astype(int))
    assert (edges == 255).sum() > 0, "Failed to detect edges!"
    print("[OK] Canny edge detection verified successfully!")
