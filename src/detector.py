"""Sinh vùng biển bằng xử lý ảnh và lọc bằng SVM tuyến tính."""

import json
from pathlib import Path

import cv2
import numpy as np

from src.preprocess import crop_plate


def iou(a, b):
    overlap_w = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    overlap_h = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    intersection = overlap_w * overlap_h
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - intersection
    return intersection / union if union else 0.0

def prepare_edges(gray, max_side, *, extended=False):
    source_height, source_width = gray.shape
    scale = min(1, max_side / max(gray.shape))
    if scale < 1:
        size = (max(1, round(source_width * scale)), max(1, round(source_height * scale)))
        gray = cv2.resize(gray, size, interpolation=cv2.INTER_AREA)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    gradient = cv2.Sobel(blur, cv2.CV_32F, 1, 0, ksize=3)
    sobel = cv2.convertScaleAbs(gradient)
    _, sobel_mask = cv2.threshold(sobel, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    edges = {
        'sobel': sobel_mask,
        'canny': cv2.Canny(blur, 50, 150),
    }
    if extended:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 5))
        blackhat = cv2.morphologyEx(blur, cv2.MORPH_BLACKHAT, kernel)
        _, edges['blackhat'] = cv2.threshold(blackhat, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    return edges, (source_width / gray.shape[1], source_height / gray.shape[0])

def candidate_boxes(edges, scale, config, max_boxes):
    height, width = edges.shape
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, tuple(config['kernel']))
    mask = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    retrieval = cv2.RETR_LIST if config.get('nested') else cv2.RETR_EXTERNAL
    contours, _ = cv2.findContours(mask, retrieval, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = w * h
        if w < 14 or h < 8 or not config.get('min_aspect', 1.15) <= w / h <= 6.8:
            continue
        if not config['min_area_ratio'] <= area / (width * height) <= 0.35:
            continue
        fill = cv2.contourArea(contour) / area
        edge_density = cv2.countNonZero(edges[y:y+h, x:x+w]) / area
        if fill < config.get('min_fill', 0.25) or edge_density < config.get('min_density', 0.065):
            continue
        margin = max(1, round(h * 0.08))
        box = (
            max(0, x - margin) * scale[0],
            max(0, y - margin) * scale[1],
            min(width, x + w + margin) * scale[0],
            min(height, y + h + margin) * scale[1],
        )
        candidates.append((box, fill * min(edge_density, 0.35)))
    return suppress_overlaps(candidates, config.get('overlap', 0.45), max_boxes)


def suppress_overlaps(candidates, overlap, limit):
    kept = []
    for box, score in sorted(candidates, key=lambda item: -item[1]):
        if all(iou(box, other) < overlap for other, _ in kept):
            kept.append((box, score))
            if len(kept) == limit:
                break
    return kept

def propose(gray, config, *, max_side=768, max_boxes=5):
    edges, scale = prepare_edges(gray, max_side)
    return candidate_boxes(edges[config['method']], scale, config, max_boxes)


def propose_ensemble(gray, config, max_side):
    """Gom vùng từ nhiều bản đồ biên và kích thước phần tử cấu trúc."""
    edges, scale = prepare_edges(gray, max_side, extended=True)
    groups = []
    for settings in config['branches']:
        groups.extend(candidate_boxes(edges[settings['method']], scale, settings, config['per_branch']))
    return suppress_overlaps(groups, 0.88, config['max_candidates'])


def proposal_features(gray, box):
    """Histogram 9 hướng trên ô 8×8, chuẩn hóa theo khối 2×2 ô."""
    patch = cv2.resize(crop_plate(gray, box), (64, 32), interpolation=cv2.INTER_AREA)
    dx = cv2.Sobel(patch, cv2.CV_32F, 1, 0, ksize=1)
    dy = cv2.Sobel(patch, cv2.CV_32F, 0, 1, ksize=1)
    magnitude = np.hypot(dx, dy)
    orientation = np.mod(np.arctan2(dy, dx), np.pi) * (9 / np.pi)
    lower = np.floor(orientation).astype(np.intp)
    fraction = orientation - lower
    cell_ids = np.arange(32).reshape(4, 8).repeat(8, axis=0).repeat(8, axis=1) * 9
    bins = np.bincount((cell_ids + lower % 9).ravel(),
                       weights=(magnitude * (1 - fraction)).ravel(), minlength=288)
    bins += np.bincount((cell_ids + (lower + 1) % 9).ravel(),
                        weights=(magnitude * fraction).ravel(), minlength=288)
    hist = bins.reshape(4, 8, 9)
    blocks = np.concatenate((hist[:-1, :-1], hist[:-1, 1:], hist[1:, :-1], hist[1:, 1:]), axis=-1)
    blocks /= np.sqrt(np.sum(blocks ** 2, axis=-1, keepdims=True) + 1e-6)
    np.minimum(blocks, 0.2, out=blocks)
    blocks /= np.sqrt(np.sum(blocks ** 2, axis=-1, keepdims=True) + 1e-6)
    return blocks.ravel().astype(np.float32)


def score_proposals(gray, proposals, weights, intercept):
    if not proposals:
        return []
    features = np.stack([proposal_features(gray, box) for box, _ in proposals])
    scores = features @ weights + intercept
    return [(box, float(score)) for (box, _), score in zip(proposals, scores)]


def box_transform(box, target):
    width, height = box[2] - box[0], box[3] - box[1]
    target_width, target_height = target[2] - target[0], target[3] - target[1]
    return [(target[0] + target[2] - box[0] - box[2]) / (2 * width),
            (target[1] + target[3] - box[1] - box[3]) / (2 * height),
            np.log(target_width / width), np.log(target_height / height)]


def refine_box(box, transform, shape):
    height, width = shape
    x, y, log_w, log_h = np.clip(transform, -0.7, 0.7)
    w, h = box[2] - box[0], box[3] - box[1]
    center_x, center_y = (box[0] + box[2]) / 2 + x * w, (box[1] + box[3]) / 2 + y * h
    half_w, half_h = w * np.exp(log_w) / 2, h * np.exp(log_h) / 2
    refined = (max(0, center_x - half_w), max(0, center_y - half_h),
               min(width, center_x + half_w), min(height, center_y + half_h))
    return refined if refined[2] > refined[0] and refined[3] > refined[1] else box


class PlateDetector:
    """Nạp cấu hình và trọng số SVM do notebook xuất."""

    def __init__(self, artifact_path):
        artifact_path = Path(artifact_path)
        self.config = json.loads(artifact_path.read_text(encoding='utf-8'))
        with np.load(artifact_path.with_name(self.config['weights_file']), allow_pickle=False) as model:
            self.weights = model['weights']
            self.intercept = model['intercept'].item()
            self.box_weights = model['box_weights'] if 'box_weights' in model else None
            self.box_intercept = model['box_intercept'] if self.box_weights is not None else None

    def predict(self, gray):
        if 'ensemble' in self.config:
            proposals = propose_ensemble(gray, self.config['ensemble'], self.config['max_side'])
        else:
            proposals = propose(gray, self.config['proposal'],
                                max_side=self.config['max_side'], max_boxes=self.config['max_boxes'])
        if self.box_weights is not None and proposals:
            features = np.stack([proposal_features(gray, box) for box, _ in proposals])
            scores = features @ self.weights + self.intercept
            transforms = (features @ self.box_weights + self.box_intercept) * self.config['box_blend']
            scored = [(refine_box(box, transform, gray.shape), float(score))
                      for (box, _), transform, score in zip(proposals, transforms, scores)]
        else:
            scored = score_proposals(gray, proposals, self.weights, self.intercept)
        filtered = [(box, score) for box, score in scored if score >= self.config['threshold']]
        if 'ensemble' in self.config:
            return suppress_overlaps(filtered, self.config['nms_iou'], self.config['max_boxes'])
        return filtered
