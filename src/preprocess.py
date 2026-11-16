"""Cắt vùng biển và tạo mask ký tự."""

import cv2


def crop_plate(gray, box):
    height, width = gray.shape
    x1, y1, x2, y2 = map(round, box)
    crop = gray[max(0, y1):min(height, y2), max(0, x1):min(width, x2)]
    if not crop.size:
        raise ValueError('Khung biển số rỗng.')
    return crop


def binarize_plate(crop, method='otsu'):
    """Đưa crop về cùng thang chiều cao, với nét chữ sáng trên nền tối."""
    height, width = crop.shape
    resized = cv2.resize(crop, (max(1, round(width * 96 / height)), 96))
    blurred = cv2.GaussianBlur(resized, (3, 3), 0)
    if method == 'otsu':
        _, mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    elif method == 'adaptive':
        mask = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 7
        )
    else:
        raise ValueError(f'Phương pháp nhị phân hóa không hợp lệ: {method}')

    if cv2.countNonZero(mask) > mask.size / 2:
        mask = cv2.bitwise_not(mask)
    return mask
