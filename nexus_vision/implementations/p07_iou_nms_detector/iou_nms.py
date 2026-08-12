"""Vectorized IoU & Non-Maximum Suppression (NMS) Algorithms from First Principles.

Implements:
1. Bounding Box Coordinate Format Transformations ([x1, y1, x2, y2] <-> [cx, cy, w, h])
2. Vectorized Pairwise Intersection over Union (IoU) Matrix (M x N)
3. Greedy Hard Non-Maximum Suppression (Hard-NMS)
4. Soft-NMS (Gaussian penalty decay for densely overlapping instances)
"""

import numpy as np
from typing import Tuple, List, Union

def box_xywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    """Converts bounding boxes from [center_x, center_y, width, height] to [x1, y1, x2, y2]."""
    cx, cy, w, h = boxes[..., 0], boxes[..., 1], boxes[..., 2], boxes[..., 3]
    x1 = cx - 0.5 * w
    y1 = cy - 0.5 * h
    x2 = cx + 0.5 * w
    y2 = cy + 0.5 * h
    return np.stack([x1, y1, x2, y2], axis=-1)

def box_xyxy_to_xywh(boxes: np.ndarray) -> np.ndarray:
    """Converts bounding boxes from [x1, y1, x2, y2] to [center_x, center_y, width, height]."""
    x1, y1, x2, y2 = boxes[..., 0], boxes[..., 1], boxes[..., 2], boxes[..., 3]
    cx = (x1 + x2) * 0.5
    cy = (y1 + y2) * 0.5
    w = x2 - x1
    h = y2 - y1
    return np.stack([cx, cy, w, h], axis=-1)

def compute_pairwise_iou(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """Computes pairwise Intersection over Union (IoU) between two sets of boxes.

    Args:
        boxes_a: Array of shape (M, 4) in [x1, y1, x2, y2]
        boxes_b: Array of shape (N, 4) in [x1, y1, x2, y2]

    Returns:
        iou: Array of shape (M, N) with IoU values in [0.0, 1.0]
    """
    # Expand dims for vectorized broadcasting: (M, 1, 4) vs (1, N, 4)
    A = np.expand_dims(boxes_a, 1)
    B = np.expand_dims(boxes_b, 0)

    # Intersection coordinates
    inter_x1 = np.maximum(A[..., 0], B[..., 0])
    inter_y1 = np.maximum(A[..., 1], B[..., 1])
    inter_x2 = np.minimum(A[..., 2], B[..., 2])
    inter_y2 = np.minimum(A[..., 3], B[..., 3])

    inter_w = np.maximum(0.0, inter_x2 - inter_x1)
    inter_h = np.maximum(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = np.maximum(0.0, boxes_a[:, 2] - boxes_a[:, 0]) * np.maximum(0.0, boxes_a[:, 3] - boxes_a[:, 1])
    area_b = np.maximum(0.0, boxes_b[:, 2] - boxes_b[:, 0]) * np.maximum(0.0, boxes_b[:, 3] - boxes_b[:, 1])

    union_area = np.expand_dims(area_a, 1) + np.expand_dims(area_b, 0) - inter_area
    iou = inter_area / np.maximum(union_area, 1e-12)
    return iou

def non_maximum_suppression(
    boxes: np.ndarray, 
    scores: np.ndarray, 
    iou_threshold: float = 0.5, 
    score_threshold: float = 0.05
) -> List[int]:
    """Greedy Non-Maximum Suppression (Hard-NMS).

    Args:
        boxes: Array of shape (N, 4) in [x1, y1, x2, y2]
        scores: Array of shape (N,) containing class confidence scores
        iou_threshold: Overlap threshold above which redundant boxes are suppressed
        score_threshold: Minimum confidence score to consider

    Returns:
        keep: List of indices of preserved bounding boxes
    """
    valid_mask = scores >= score_threshold
    boxes = boxes[valid_mask]
    scores = scores[valid_mask]
    original_indices = np.where(valid_mask)[0]

    if len(boxes) == 0:
        return []

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)

    order = scores.argsort()[::-1]
    keep = []

    while order.size > 0:
        i = order[0]
        keep.append(original_indices[i])

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-12)

        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]

    return keep

def soft_nms(
    boxes: np.ndarray, 
    scores: np.ndarray, 
    sigma: float = 0.5, 
    score_threshold: float = 0.01
) -> Tuple[np.ndarray, np.ndarray]:
    """Soft-NMS with Gaussian score decay penalty (Bodla et al., 2017)."""
    N = boxes.shape[0]
    boxes_out = boxes.copy()
    scores_out = scores.copy()
    
    for i in range(N):
        # Find maximum score in remaining subset
        max_idx = i + np.argmax(scores_out[i:])
        # Swap current box with max box
        boxes_out[[i, max_idx]] = boxes_out[[max_idx, i]]
        scores_out[[i, max_idx]] = scores_out[[max_idx, i]]
        
        # Compute IoU of remaining boxes with current box
        current_box = boxes_out[i:i+1]
        remaining_boxes = boxes_out[i+1:]
        
        if len(remaining_boxes) > 0:
            ious = compute_pairwise_iou(current_box, remaining_boxes)[0]
            # Gaussian decay weighting: score = score * exp(-iou^2 / sigma)
            decay = np.exp(-(ious ** 2) / sigma)
            scores_out[i+1:] = scores_out[i+1:] * decay
            
    keep = scores_out >= score_threshold
    return boxes_out[keep], scores_out[keep]

if __name__ == "__main__":
    print("=== IoU & NMS Scratch Implementation Demo ===")
    
    # Define overlapping candidate bounding boxes
    boxes = np.array([
        [10, 10, 50, 50],   # Box 0 (Highest score, true detection)
        [12, 11, 49, 52],   # Box 1 (High overlap with Box 0, redundant)
        [100, 100, 150, 150]# Box 2 (Separate distant object)
    ], dtype=np.float32)
    scores = np.array([0.95, 0.88, 0.90], dtype=np.float32)
    
    # Compute pairwise IoU
    iou_mat = compute_pairwise_iou(boxes, boxes)
    print("Pairwise IoU Matrix:\n", iou_mat.round(3))
    
    # Execute NMS
    kept_indices = non_maximum_suppression(boxes, scores, iou_threshold=0.5)
    print(f"\nInitial Candidates: {len(boxes)} boxes")
    print(f"Kept Bounding Box Indices after NMS: {kept_indices}")
    assert kept_indices == [0, 2], f"Expected indices [0, 2], got {kept_indices}"
    print("[OK] Vectorized IoU & NMS verified successfully!")
