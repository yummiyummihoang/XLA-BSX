"""Xuất crop theo khung nhãn và các ứng viên ký tự để duyệt thủ công."""

import argparse
import csv
import random
from pathlib import Path

import cv2
import numpy as np

from src.preprocess import binarize_plate, crop_plate
from src.segmentation import character_rows


PROJECT = Path(__file__).resolve().parents[1]
DATASET_NAME = 'License Plate Detection Dataset'
EXCLUDED = {'train': {'quandoi64.jpg', 'quandoi92.jpg', 'quandoi99.jpg'}, 'val': set()}


def save_png(path, image):
    ok, encoded = cv2.imencode('.png', image)
    if not ok:
        raise RuntimeError(f'Không ghi được {path}')
    encoded.tofile(path)


def read_gray(path):
    gray = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise ValueError(f'Không đọc được {path}')
    return gray


def load_boxes(path, shape):
    height, width = shape
    boxes = []
    for line in path.read_text(encoding='utf-8-sig').splitlines():
        if not line.strip():
            continue
        label, x, y, w, h = map(float, line.split())
        if label != 0 or w <= 0 or h <= 0:
            raise ValueError(f'Nhãn không hợp lệ: {path}')
        boxes.append(((x - w / 2) * width, (y - h / 2) * height,
                      (x + w / 2) * width, (y + h / 2) * height))
    return boxes


def write_csv(path, fields, rows):
    with path.open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(fields)
        writer.writerows(rows)


def export_review(root, output, split, limit, method):
    image_dir = root / 'images' / split
    sources = sorted(path for path in image_dir.iterdir()
                     if path.suffix.lower() in {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}
                     and path.name not in EXCLUDED[split])
    if not sources:
        raise ValueError(f'Không có ảnh tại {image_dir}')
    random.Random(42).shuffle(sources)
    output.mkdir(parents=True, exist_ok=False)
    plate_records, glyph_records = [], []
    total_candidates = 0
    for source in sources:
        gray = read_gray(source)
        boxes = load_boxes(root / 'labels' / split / source.with_suffix('.txt').name, gray.shape)
        for number, box in enumerate(boxes, 1):
            plate_id = f'{source.stem}_{number}'
            relative_source = source.relative_to(root).as_posix()
            folder = output / plate_id
            folder.mkdir()
            crop = crop_plate(gray, box)
            mask = binarize_plate(crop, method)
            rows = character_rows(mask)
            preview = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            save_png(folder / 'crop.png', crop)
            save_png(folder / 'mask.png', mask)
            for row_index, row in enumerate(rows, 1):
                for column_index, (x, y, w, h, glyph) in enumerate(row, 1):
                    name = f'r{row_index}_c{column_index}.png'
                    if split == 'train':
                        save_png(folder / name, glyph)
                        glyph_records.append((plate_id, relative_source, f'{plate_id}/{name}',
                                              row_index, column_index, ''))
                    cv2.rectangle(preview, (x, y), (x + w - 1, y + h - 1), (0, 255, 0), 1)
            save_png(folder / 'preview.png', preview)
            candidate_count = sum(map(len, rows))
            total_candidates += candidate_count
            plate_records.append((plate_id, relative_source, *(round(value, 2) for value in box),
                                  len(rows), candidate_count, '', ''))
            if len(plate_records) >= limit:
                break
        if len(plate_records) >= limit:
            break
    write_csv(output / 'plates.csv',
              ('plate_id', 'source_image', 'x1', 'y1', 'x2', 'y2', 'rows', 'candidates',
               'plate_text', 'readable'), plate_records)
    if split == 'train':
        write_csv(output / 'characters.csv',
                  ('plate_id', 'source_image', 'glyph_path', 'row', 'column', 'label'), glyph_records)
    print(f'{split}: {len(plate_records)} plates, {total_candidates} character candidates')
    print(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path)
    parser.add_argument('--split', choices=('train', 'val'), required=True)
    parser.add_argument('--limit', type=int, default=30)
    parser.add_argument('--method', choices=('otsu', 'adaptive'), default='otsu')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.limit < 1:
        parser.error('--limit phải lớn hơn 0.')
    roots = (PROJECT / DATASET_NAME, PROJECT.parent / 'CuoiKy' / DATASET_NAME)
    root = args.dataset or next((path for path in roots if path.is_dir()), None)
    if root is None:
        parser.error('Đặt --dataset trỏ tới thư mục có images và labels.')
    output = args.output or PROJECT / 'outputs' / 'ocr_review' / f'{args.split}_{args.method}'
    if output.exists():
        parser.error(f'{output} đã tồn tại. Dùng --output mới để giữ các nhãn đã nhập.')
    export_review(root, output, args.split, args.limit, args.method)


if __name__ == '__main__':
    main()
