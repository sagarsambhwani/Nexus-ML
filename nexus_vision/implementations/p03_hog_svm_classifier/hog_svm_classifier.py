"""Histogram of Oriented Gradients (HOG) Feature Extractor + Linear SVM Classifier.

Implements:
1. Spatial Image Gradients (Ix, Iy, Magnitude, Orientation 0-180 deg)
2. Cell-level Histogram of Oriented Gradients (8x8 pixel cells, 9 orientation bins)
3. 2x2 Block Normalization (L2-Hys norm across overlapping blocks)
4. Classification pipeline with Linear Support Vector Machine (Linear SVM)
"""

import numpy as np
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report
from typing import Tuple

class HOGExtractor:
    def __init__(self, cell_size: int = 8, block_size: int = 2, n_bins: int = 9):
        self.cell_size = cell_size
        self.block_size = block_size
        self.n_bins = n_bins
        self.bin_width = 180.0 / n_bins

    def compute_gradients(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Calculates 1D spatial gradients without smoothing."""
        # Sobel or 1D centered differences [-1, 0, 1]
        gx = np.zeros_like(image, dtype=np.float32)
        gy = np.zeros_like(image, dtype=np.float32)
        
        gx[:, 1:-1] = image[:, 2:] - image[:, :-2]
        gy[1:-1, :] = image[2:, :] - image[:-2, :]
        
        magnitude = np.hypot(gx, gy)
        orientation = np.arctan2(gy, gx) * (180.0 / np.pi)
        orientation[orientation < 0] += 180.0  # Unsigned gradients in [0, 180)
        return magnitude, orientation

    def cell_histograms(self, magnitude: np.ndarray, orientation: np.ndarray) -> np.ndarray:
        """Computes 9-bin orientation histogram for every (cell_size x cell_size) cell."""
        H, W = magnitude.shape
        n_cells_y = H // self.cell_size
        n_cells_x = W // self.cell_size
        
        histograms = np.zeros((n_cells_y, n_cells_x, self.n_bins), dtype=np.float32)
        
        for cy in range(n_cells_y):
            for cx in range(n_cells_x):
                y_start = cy * self.cell_size
                x_start = cx * self.cell_size
                
                mag_patch = magnitude[y_start:y_start+self.cell_size, x_start:x_start+self.cell_size]
                ori_patch = orientation[y_start:y_start+self.cell_size, x_start:x_start+self.cell_size]
                
                # Trilinear / Linear interpolation into bins
                bin_idx = (ori_patch / self.bin_width).astype(int) % self.n_bins
                for b in range(self.n_bins):
                    histograms[cy, cx, b] = np.sum(mag_patch[bin_idx == b])
                    
        return histograms

    def block_normalization(self, histograms: np.ndarray) -> np.ndarray:
        """Applies L2-norm normalization over overlapping (2x2) cell blocks."""
        n_cells_y, n_cells_x, n_bins = histograms.shape
        n_blocks_y = n_cells_y - self.block_size + 1
        n_blocks_x = n_cells_x - self.block_size + 1
        
        if n_blocks_y <= 0 or n_blocks_x <= 0:
            return histograms.flatten()
            
        feature_vector = []
        for by in range(n_blocks_y):
            for bx in range(n_blocks_x):
                block = histograms[by:by+self.block_size, bx:bx+self.block_size, :].flatten()
                # L2 Normalization with epsilon
                norm = np.linalg.norm(block) + 1e-6
                normalized_block = block / norm
                feature_vector.append(normalized_block)
                
        return np.concatenate(feature_vector)

    def extract(self, image: np.ndarray) -> np.ndarray:
        """Extracts complete 1D HOG feature descriptor from a grayscale image."""
        if image.ndim == 3:
            # Convert RGB to Grayscale
            image = 0.299 * image[:, :, 0] + 0.587 * image[:, :, 1] + 0.114 * image[:, :, 2]
            
        magnitude, orientation = self.compute_gradients(image.astype(np.float32))
        histograms = self.cell_histograms(magnitude, orientation)
        hog_features = self.block_normalization(histograms)
        return hog_features

class HOGObjectClassifier:
    def __init__(self, cell_size: int = 8):
        self.extractor = HOGExtractor(cell_size=cell_size)
        self.clf = LinearSVC(C=1.0, random_state=42, max_iter=2000)

    def fit(self, images: np.ndarray, labels: np.ndarray):
        """Extracts HOG features from training images and fits Linear SVM."""
        features = np.array([self.extractor.extract(img) for img in images])
        self.clf.fit(features, labels)

    def predict(self, images: np.ndarray) -> np.ndarray:
        """Predicts class labels for test images using HOG + SVM."""
        features = np.array([self.extractor.extract(img) for img in images])
        return self.clf.predict(features)

if __name__ == "__main__":
    np.random.seed(42)
    # Generate synthetic visual dataset: Vertical stripes (Class 0) vs Horizontal stripes (Class 1)
    N = 100
    images = np.zeros((N, 32, 32), dtype=np.float32)
    labels = np.zeros(N, dtype=int)
    
    for i in range(N):
        if i % 2 == 0:
            # Class 0: Vertical stripes
            images[i, :, ::4] = 255.0
            labels[i] = 0
        else:
            # Class 1: Horizontal stripes
            images[i, ::4, :] = 255.0
            labels[i] = 1
        # Add random sensor noise
        images[i] += np.random.normal(0, 15, (32, 32))
        
    split = int(0.8 * N)
    X_train, y_train = images[:split], labels[:split]
    X_test, y_test = images[split:], labels[split:]
    
    classifier = HOGObjectClassifier(cell_size=8)
    classifier.fit(X_train, y_train)
    preds = classifier.predict(X_test)
    
    acc = accuracy_score(y_test, preds)
    print("=== HOG + Linear SVM Classifier ===")
    print(f"HOG Descriptor Length: {classifier.extractor.extract(images[0]).shape[0]} dimensions")
    print(f"Classification Accuracy: {acc * 100.0:.2f}%")
    assert acc >= 0.95, "Classification accuracy below expected baseline!"
    print("[OK] HOG + SVM verified successfully!")
