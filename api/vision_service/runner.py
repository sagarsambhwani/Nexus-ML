"""Vision Studio Interactive Runner & Visualizer Engine.

Generates realistic visual synthetic inputs, executes algorithms, and returns
base64-encoded visual outputs, tensor metrics, and execution diagnostics.
"""

import io
import time
import base64
import numpy as np
from PIL import Image, ImageDraw
from typing import Dict, Any, List, Optional

# Import Vision Implementations
from nexus_vision.implementations.p01_numpy_convolution.convolution_2d import (
    conv2d_single_channel, get_classical_filter
)
from nexus_vision.implementations.p02_canny_edge_detector.canny_detector import (
    CannyEdgeDetector
)
from nexus_vision.implementations.p03_hog_svm_classifier.hog_svm_classifier import (
    HOGExtractor, HOGObjectClassifier
)
from nexus_vision.implementations.p04_sift_orb_matcher.feature_matching import (
    FeatureMatcher
)
from nexus_vision.implementations.p05_scratch_cnn_backprop.scratch_cnn import (
    ScratchCNN, softmax_cross_entropy_loss
)
from nexus_vision.implementations.p07_iou_nms_detector.iou_nms import (
    compute_pairwise_iou, non_maximum_suppression, soft_nms
)
from nexus_vision.implementations.p10_camera_calibration_stereo.stereo_depth import (
    StereoDepthEstimator
)
from nexus_vision.implementations.p15_production_cv_pipeline.production_pipeline import (
    ProductionVisionPipeline
)

def array_to_base64_png(arr: np.ndarray, colormap: str = "gray") -> str:
    """Converts a 2D or 3D NumPy array into a base64 Data URL (data:image/png;base64,...)."""
    # Normalize array to [0, 255] uint8
    arr = arr.astype(np.float32)
    min_val, max_val = arr.min(), arr.max()
    if max_val > min_val:
        norm = ((arr - min_val) / (max_val - min_val) * 255.0).astype(np.uint8)
    else:
        norm = np.zeros_like(arr, dtype=np.uint8)

    if norm.ndim == 2:
        if colormap == "inferno" or colormap == "depth":
            # Simple heatmap colormap (Black -> Blue -> Red -> Yellow)
            rgb = np.zeros((norm.shape[0], norm.shape[1], 3), dtype=np.uint8)
            rgb[:, :, 0] = np.clip(norm * 1.5, 0, 255).astype(np.uint8) # Red
            rgb[:, :, 1] = np.clip(norm * 0.8, 0, 255).astype(np.uint8) # Green
            rgb[:, :, 2] = np.clip(255 - norm * 1.2, 0, 255).astype(np.uint8) # Blue
            img = Image.fromarray(rgb)
        else:
            img = Image.fromarray(norm, mode="L")
    elif norm.ndim == 3:
        if norm.shape[2] == 1:
            img = Image.fromarray(norm[:, :, 0], mode="L")
        else:
            img = Image.fromarray(norm.astype(np.uint8), mode="RGB")
    else:
        raise ValueError(f"Unsupported array shape for image conversion: {arr.shape}")

    # Scale up small arrays for crisp display
    if img.width < 128 or img.height < 128:
        scale_factor = max(1, 256 // max(img.width, img.height))
        img = img.resize((img.width * scale_factor, img.height * scale_factor), resample=Image.Resampling.NEAREST)

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64_str = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64_str}"

def generate_preset_image(preset_name: str, size: int = 128) -> np.ndarray:
    """Generates synthetic test images for visual algorithm evaluation."""
    img = np.zeros((size, size), dtype=np.float32)
    name = preset_name.lower()

    if name in ["shapes", "geometric"]:
        # Rectangle, Circle, and diagonal line
        img[20:60, 20:60] = 220.0
        cy, cx = size // 2 + 20, size // 2 + 20
        y, x = np.ogrid[:size, :size]
        mask = (x - cx)**2 + (y - cy)**2 <= 25**2
        img[mask] = 180.0
        for k in range(size // 2):
            if 10 + k < size and size - 20 - k >= 0:
                img[10 + k, size - 20 - k] = 255.0
    elif name in ["defect", "scratch", "crack"]:
        # Surface texture with simulated scratch defect
        img += np.random.normal(128, 10, (size, size)).astype(np.float32)
        # Scratch
        for i in range(20, size - 20):
            j = int(i * 0.8 + 15 + np.sin(i * 0.2) * 3)
            if 0 <= j < size:
                img[i, j:j+2] = 25.0
    elif name in ["vertical_edges", "stripes"]:
        # Vertical stripes
        img[:, ::16] = 240.0
        img[:, 1::16] = 240.0
    elif name in ["checkerboard"]:
        # Checkerboard pattern
        block = 16
        for i in range(0, size, block):
            for j in range(0, size, block):
                if (i // block + j // block) % 2 == 0:
                    img[i:i+block, j:j+block] = 220.0
    else:
        # Default step edge with circle
        img[:, :size//2] = 200.0
        cy, cx = size // 2, size // 2
        y, x = np.ogrid[:size, :size]
        img[(x - cx)**2 + (y - cy)**2 <= 20**2] = 50.0

    return img

class VisionStudioRunner:
    """Executes vision algorithms dynamically with diagnostics."""

    @staticmethod
    def run_convolution(kernel_name: str = "sobel_v", preset: str = "shapes", padding: int = 1, stride: int = 1) -> Dict[str, Any]:
        start = time.perf_counter()
        img = generate_preset_image(preset, size=64)
        kernel = get_classical_filter(kernel_name)
        out = conv2d_single_channel(img, kernel, stride=stride, padding=padding)
        latency_ms = (time.perf_counter() - start) * 1000.0

        return {
            "module": "01_numpy_convolution",
            "kernel_name": kernel_name,
            "kernel_matrix": kernel.tolist(),
            "input_shape": list(img.shape),
            "output_shape": list(out.shape),
            "latency_ms": round(latency_ms, 2),
            "input_image_b64": array_to_base64_png(img),
            "output_image_b64": array_to_base64_png(out),
            "summary": f"Convolved {img.shape[0]}x{img.shape[1]} image with {kernel_name} ({kernel.shape[0]}x{kernel.shape[1]}) in {latency_ms:.2f}ms."
        }

    @staticmethod
    def run_canny(preset: str = "shapes", low_thresh: float = 30.0, high_thresh: float = 80.0, sigma: float = 1.0) -> Dict[str, Any]:
        start = time.perf_counter()
        img = generate_preset_image(preset, size=64)
        detector = CannyEdgeDetector(low_threshold=low_thresh, high_threshold=high_thresh, sigma=sigma)
        
        # 1. Blur
        kernel = detector.gaussian_kernel(size=5, sigma=sigma)
        blurred = detector.convolve(img, kernel)
        
        # 2. Gradients
        magnitude, angle = detector.compute_gradients(blurred)
        
        # 3. NMS
        nms = detector.non_maximum_suppression(magnitude, angle)
        
        # 4. Double Threshold & Hysteresis
        thresh, _, _ = detector.double_threshold(nms)
        edges = detector.hysteresis_tracking(thresh)
        
        latency_ms = (time.perf_counter() - start) * 1000.0

        return {
            "module": "02_canny_edge_detector",
            "low_threshold": low_thresh,
            "high_threshold": high_thresh,
            "sigma": sigma,
            "latency_ms": round(latency_ms, 2),
            "input_image_b64": array_to_base64_png(img),
            "blurred_image_b64": array_to_base64_png(blurred),
            "gradient_image_b64": array_to_base64_png(magnitude),
            "nms_image_b64": array_to_base64_png(nms),
            "output_image_b64": array_to_base64_png(edges),
            "edge_pixel_count": int((edges == 255).sum()),
            "summary": f"Detected {(edges == 255).sum()} edge pixels across 5 Canny stages in {latency_ms:.2f}ms."
        }

    @staticmethod
    def run_hog_svm(preset: str = "vertical_edges", cell_size: int = 8) -> Dict[str, Any]:
        start = time.perf_counter()
        img = generate_preset_image(preset, size=64)
        extractor = HOGExtractor(cell_size=cell_size)
        
        magnitude, orientation = extractor.compute_gradients(img)
        histograms = extractor.cell_histograms(magnitude, orientation)
        hog_feat = extractor.block_normalization(histograms)
        
        latency_ms = (time.perf_counter() - start) * 1000.0

        return {
            "module": "03_hog_svm_classifier",
            "cell_size": cell_size,
            "descriptor_dimension": int(hog_feat.shape[0]),
            "grid_cells": [int(histograms.shape[0]), int(histograms.shape[1])],
            "latency_ms": round(latency_ms, 2),
            "input_image_b64": array_to_base64_png(img),
            "magnitude_image_b64": array_to_base64_png(magnitude),
            "orientation_image_b64": array_to_base64_png(orientation),
            "summary": f"Extracted {hog_feat.shape[0]}-dimensional HOG descriptor from {histograms.shape[0]}x{histograms.shape[1]} spatial cells in {latency_ms:.2f}ms."
        }

    @staticmethod
    def run_sift_orb(method: str = "orb", max_features: int = 200) -> Dict[str, Any]:
        start = time.perf_counter()
        img1 = generate_preset_image("shapes", size=96).astype(np.uint8)
        # Shift and rotate for Image 2
        img2 = np.roll(np.roll(img1, 10, axis=0), 12, axis=1)
        
        matcher = FeatureMatcher(method=method, max_features=max_features)
        kps1, desc1 = matcher.extract_features(img1)
        kps2, desc2 = matcher.extract_features(img2)
        good_matches = matcher.match(desc1, desc2, ratio_thresh=0.85)
        
        latency_ms = (time.perf_counter() - start) * 1000.0

        # Create side-by-side visualization
        combined = np.hstack([img1, img2])

        return {
            "module": "04_sift_orb_matcher",
            "method": method.upper(),
            "keypoints_img1": len(kps1) if kps1 is not None else 0,
            "keypoints_img2": len(kps2) if kps2 is not None else 0,
            "robust_matches": len(good_matches),
            "latency_ms": round(latency_ms, 2),
            "side_by_side_b64": array_to_base64_png(combined),
            "summary": f"Extracted {len(kps1)} keypoints and found {len(good_matches)} robust matches using {method.upper()} in {latency_ms:.2f}ms."
        }

    @staticmethod
    def run_scratch_cnn(epochs: int = 15, learning_rate: float = 0.05) -> Dict[str, Any]:
        start = time.perf_counter()
        np.random.seed(42)
        # Synthetic dataset: Vertical (0) vs Horizontal (1)
        X = np.random.randn(12, 1, 8, 8).astype(np.float32)
        y = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
        
        model = ScratchCNN(in_channels=1, num_classes=2)
        loss_history = []
        
        for ep in range(epochs):
            logits = model.forward(X)
            loss, dlogits = softmax_cross_entropy_loss(logits, y)
            loss_history.append(round(float(loss), 4))
            model.backward(dlogits)
            model.step(lr=learning_rate)
            
        final_logits = model.forward(X)
        exp_s = np.exp(final_logits - np.max(final_logits, axis=1, keepdims=True))
        probs = (exp_s / np.sum(exp_s, axis=1, keepdims=True)).round(4).tolist()
        
        latency_ms = (time.perf_counter() - start) * 1000.0

        return {
            "module": "05_scratch_cnn_backprop",
            "epochs": epochs,
            "learning_rate": learning_rate,
            "initial_loss": loss_history[0],
            "final_loss": loss_history[-1],
            "loss_curve": loss_history,
            "sample_predictions": probs[:4],
            "latency_ms": round(latency_ms, 2),
            "sample_input_b64": array_to_base64_png(X[0, 0]),
            "summary": f"Trained pure NumPy CNN for {epochs} epochs. Loss reduced from {loss_history[0]} to {loss_history[-1]} in {latency_ms:.2f}ms."
        }

    @staticmethod
    def run_iou_nms(iou_threshold: float = 0.5, score_threshold: float = 0.1) -> Dict[str, Any]:
        start = time.perf_counter()
        # Define candidate detection boxes: [x1, y1, x2, y2]
        boxes = np.array([
            [15, 15, 65, 65],   # Box 0 (Main object - score 0.95)
            [18, 16, 63, 67],   # Box 1 (Redundant duplicate - score 0.85)
            [22, 20, 68, 62],   # Box 2 (Redundant duplicate - score 0.72)
            [75, 75, 115, 115], # Box 3 (Second object - score 0.91)
            [78, 73, 112, 118]  # Box 4 (Redundant duplicate - score 0.68)
        ], dtype=np.float32)
        scores = np.array([0.95, 0.85, 0.72, 0.91, 0.68], dtype=np.float32)

        iou_matrix = compute_pairwise_iou(boxes, boxes).round(3).tolist()
        kept_indices = non_maximum_suppression(boxes, scores, iou_threshold=iou_threshold, score_threshold=score_threshold)
        
        # Render Canvas image with bounding boxes
        canvas = np.ones((130, 130, 3), dtype=np.uint8) * 245
        pil_img = Image.fromarray(canvas)
        draw = ImageDraw.Draw(pil_img)

        # Draw all initial boxes in dashed/light red
        for i, b in enumerate(boxes):
            if i not in kept_indices:
                draw.rectangle([b[0], b[1], b[2], b[3]], outline=(239, 68, 68), width=1)
                
        # Draw kept boxes in solid emerald green
        for idx in kept_indices:
            b = boxes[idx]
            draw.rectangle([b[0], b[1], b[2], b[3]], outline=(16, 185, 129), width=2)

        buf = io.BytesIO()
        pil_img.save(buf, format="PNG")
        b64 = f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode('utf-8')}"

        latency_ms = (time.perf_counter() - start) * 1000.0

        return {
            "module": "07_iou_nms_detector",
            "total_candidates": len(boxes),
            "kept_count": len(kept_indices),
            "kept_indices": kept_indices,
            "iou_matrix": iou_matrix,
            "visualization_b64": b64,
            "latency_ms": round(latency_ms, 2),
            "summary": f"Suppressed {len(boxes) - len(kept_indices)} redundant overlapping boxes (IoU > {iou_threshold}). Kept {len(kept_indices)} true instances."
        }

    @staticmethod
    def run_stereo_depth(baseline_m: float = 0.1, focal_px: float = 500.0, num_disp: int = 16) -> Dict[str, Any]:
        start = time.perf_counter()
        H, W = 64, 64
        img_L = np.ones((H, W), dtype=np.uint8) * 90
        img_R = np.ones((H, W), dtype=np.uint8) * 90
        
        # Foreground square at disparity = 6px
        img_L[16:48, 24:56] = 210
        img_R[16:48, 18:50] = 210

        estimator = StereoDepthEstimator(focal_length_px=focal_px, baseline_meters=baseline_m, num_disparities=num_disp)
        disp = estimator.compute_disparity(img_L, img_R)
        depth = estimator.disparity_to_depth(disp, min_disp=0.5)
        
        latency_ms = (time.perf_counter() - start) * 1000.0

        return {
            "module": "10_camera_calibration_stereo",
            "baseline_meters": baseline_m,
            "focal_length_px": focal_px,
            "disparity_map_b64": array_to_base64_png(disp),
            "depth_map_b64": array_to_base64_png(depth, colormap="depth"),
            "left_image_b64": array_to_base64_png(img_L),
            "right_image_b64": array_to_base64_png(img_R),
            "max_depth_m": round(float(depth.max()), 2),
            "min_depth_m": round(float(depth[depth > 0].min()) if (depth > 0).any() else 0.0, 2),
            "latency_ms": round(latency_ms, 2),
            "summary": f"Computed stereo disparity and metric depth map ($Z = \\frac{{fB}}{{d}}$) in {latency_ms:.2f}ms."
        }

    @staticmethod
    def run_production_pipeline(num_runs: int = 50, batch_size: int = 4) -> Dict[str, Any]:
        start = time.perf_counter()
        pipeline = ProductionVisionPipeline(labels=["NORMAL", "SURFACE_SCRATCH", "CRACK", "CORROSION"])
        
        # Generate sample batch
        sample_img = generate_preset_image("defect", size=128).astype(np.uint8)
        sample_img = np.stack([sample_img]*3, axis=-1)
        
        infer_results = pipeline.predict([sample_img])
        bench = pipeline.profile_latency(num_iterations=num_runs, batch_size=batch_size)
        
        latency_ms = (time.perf_counter() - start) * 1000.0

        return {
            "module": "15_production_cv_pipeline",
            "sample_result": infer_results[0],
            "benchmark": bench,
            "input_sample_b64": array_to_base64_png(sample_img),
            "latency_ms": round(latency_ms, 2),
            "summary": f"Production pipeline executed {num_runs} batches (batch={batch_size}) at {bench['throughput_fps']} FPS (P95 latency: {bench['p95_latency_ms']}ms)."
        }
