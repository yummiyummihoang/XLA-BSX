"""Tách dòng và ký tự từ ảnh biển đã nhị phân hóa."""

import cv2
import numpy as np


def character_rows(mask):
    """Tìm ứng viên theo thành phần liên thông và xếp thứ tự đọc."""
    height, width = mask.shape
    _, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    boxes = []
    for index, (x, y, w, h, area) in enumerate(stats[1:], 1):
        if not 0.18 * height <= h <= 0.93 * height:
            continue
        if not 0.10 <= w / h <= 1.15 or w >= width / 2 or area < 0.10 * w * h:
            continue
        glyph = (labels[y:y + h, x:x + w] == index).astype(np.uint8) * 255
        boxes.append((int(x), int(y), int(w), int(h), glyph))

    if not boxes:
        return []
    by_height = sorted(boxes, key=lambda box: box[1] + box[3] / 2)
    centers = [box[1] + box[3] / 2 for box in by_height]
    split = int(np.argmax(np.diff(centers))) + 1 if len(centers) >= 2 else 0
    if split >= 2 and len(boxes) - split >= 2 and centers[split] - centers[split - 1] > 0.22 * height:
        rows = [by_height[:split], by_height[split:]]
    else:
        rows = [boxes]
    return [sorted(row, key=lambda box: box[0]) for row in rows]
