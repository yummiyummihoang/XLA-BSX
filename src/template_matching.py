"""So khớp ký tự với các mẫu tự tạo từ tập train."""

import csv
import re

import cv2
import numpy as np


def normalize_glyph(glyph):
    canvas = np.zeros((40, 28), dtype=np.float32)
    height, width = glyph.shape
    scale = min(36 / height, 24 / width)
    resized = cv2.resize(
        glyph, (max(1, round(width * scale)), max(1, round(height * scale))),
        interpolation=cv2.INTER_NEAREST,
    )
    h, w = resized.shape
    y, x = (40 - h) // 2, (28 - w) // 2
    canvas[y:y + h, x:x + w] = resized > 0
    return canvas


def match_glyph(glyph, templates, *, max_error, min_margin):
    """Trả về ký tự và MSE. Templates là các cặp nhãn và mẫu đã chuẩn hóa."""
    normalized = normalize_glyph(glyph)
    errors = {}
    for label, sample in templates:
        error = float(np.mean((sample - normalized) ** 2))
        errors[label] = min(errors.get(label, float('inf')), error)
    if len(errors) < 2:
        raise ValueError('Cần bộ mẫu có ít nhất hai lớp ký tự.')
    (label, error), (_, runner_up) = sorted(errors.items(), key=lambda item: item[1])[:2]
    if error > max_error or runner_up - error < min_margin:
        return '?', error
    return label, error


def load_train_templates(csv_path):
    """Nạp các ứng viên train đã được duyệt và có nhãn."""
    templates = []
    with csv_path.open(encoding='utf-8-sig', newline='') as handle:
        for row in csv.DictReader(handle):
            label = row['label'].strip().upper()
            if not label:
                continue
            if not row['source_image'].startswith('images/train/') or not re.fullmatch('[A-Z0-9]', label):
                raise ValueError(f"Mẫu không hợp lệ: {row['glyph_path']}")
            path = csv_path.parent / row['glyph_path']
            glyph = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
            if glyph is None:
                raise ValueError(f'Không đọc được mẫu: {path}')
            templates.append((label, normalize_glyph(glyph)))
    if len({label for label, _ in templates}) < 2:
        raise ValueError('Cần gán nhãn mẫu train cho ít nhất hai lớp ký tự.')
    return templates


def match_rows(rows, templates, *, max_error, min_margin):
    """Đọc theo thứ tự dòng và giữ dấu hỏi ở các ký tự chưa chắc chắn."""
    lines = []
    errors = []
    for row in rows:
        characters = []
        for _, _, _, _, glyph in row:
            label, error = match_glyph(glyph, templates, max_error=max_error, min_margin=min_margin)
            characters.append(label)
            errors.append(error)
        lines.append(''.join(characters))
    return lines, errors
