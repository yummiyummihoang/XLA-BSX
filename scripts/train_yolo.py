"""Train YOLOv8n từ kiến trúc YAML và đánh giá trên cùng split với SVM."""

import argparse
import csv
import json
import os
import shutil
from pathlib import Path

# Keep Ultralytics settings within the project workspace.
PROJECT = Path(__file__).resolve().parents[1]
config_home = PROJECT / 'outputs' / 'ultralytics_config'
config_home.mkdir(parents=True, exist_ok=True)
os.environ.setdefault('YOLO_CONFIG_DIR', str(config_home))

import numpy as np
import torch
from torchvision.ops import nms
import yaml
from ultralytics import YOLO

from src.detector import iou


EXCLUDED = {
    'train': {'quandoi64.jpg', 'quandoi92.jpg', 'quandoi99.jpg'},
    'val': set(),
    'test': {'quandoi75.jpg'},
}
IMAGE_SUFFIXES = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}


def prepare_dataset(root, output):
    # Ultralytics can repair JPEGs in place and writes label caches beside labels.
    staging = output / 'dataset'
    manifests = {}
    for split, excluded in EXCLUDED.items():
        paths = sorted(path for path in (root / 'images' / split).iterdir()
                       if path.suffix.lower() in IMAGE_SUFFIXES and path.name not in excluded)
        if not paths:
            raise ValueError(f'Split rong: {split}')
        images = staging / 'images' / split
        labels = staging / 'labels' / split
        images.mkdir(parents=True, exist_ok=True)
        labels.mkdir(parents=True, exist_ok=True)
        for path in paths:
            shutil.copyfile(path, images / path.name)
            label = path.with_suffix('.txt').name
            shutil.copyfile(root / 'labels' / split / label, labels / label)
        manifest = output / f'{split}.txt'
        manifest.write_text('\n'.join((images / path.name).as_posix() for path in paths) + '\n',
                            encoding='utf-8')
        manifests[split] = manifest.as_posix()
    data = output / 'plates.yaml'
    data.write_text(yaml.safe_dump(dict(path=output.as_posix(), **manifests, names={0: 'plate'})),
                    encoding='utf-8')
    return data, manifests


def truth_boxes(path, shape):
    height, width = shape
    labels = path.parents[2] / 'labels' / path.parent.name / path.with_suffix('.txt').name
    boxes = []
    for row in labels.read_text(encoding='utf-8-sig').splitlines():
        if not row.strip():
            continue
        label, x, y, w, h = map(float, row.split())
        if label != 0:
            raise ValueError(f'Nhan ngoai lop plate: {labels}')
        boxes.append((max(0, (x - w / 2) * width), max(0, (y - h / 2) * height),
                      min(width, (x + w / 2) * width), min(height, (y + h / 2) * height)))
    return boxes


def predict_split(model, manifest, imgsz, device):
    paths = Path(manifest).read_text(encoding='utf-8').splitlines()
    for start in range(0, len(paths), 16):
        results = model.predict(paths[start:start + 16], imgsz=imgsz, device=device, conf=0.01,
                                iou=0.45, max_det=5, verbose=False, stream=True)
        for result in results:
            truth = truth_boxes(Path(result.path), result.orig_shape)
            yield truth, result.boxes.xyxy.cpu().tolist(), result.boxes.conf.cpu().tolist()


def measure(records, threshold):
    tp = fp = fn = 0
    for truth, boxes, scores in records:
        remaining = list(truth)
        for box, score in sorted(zip(boxes, scores), key=lambda pair: -pair[1]):
            if score < threshold:
                break
            overlaps = [iou(box, target) for target in remaining]
            if overlaps and max(overlaps) >= 0.5:
                remaining.pop(int(np.argmax(overlaps)))
                tp += 1
            else:
                fp += 1
        fn += len(remaining)
    return dict(threshold=threshold, tp=tp, fp=fp, fn=fn,
                precision=tp / max(1, tp + fp), recall=tp / max(1, tp + fn),
                f1=2 * tp / max(1, 2 * tp + fp + fn))


def write_rows(path, rows):
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    default_dataset = PROJECT.parent / 'CuoiKy' / 'License Plate Detection Dataset'
    local_dataset = PROJECT / default_dataset.name
    parser.add_argument('--dataset', type=Path,
                        default=local_dataset if local_dataset.is_dir() else default_dataset)
    parser.add_argument('--output', type=Path, default=PROJECT / 'outputs' / 'yolo_scratch')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--batch', type=int, default=12)
    parser.add_argument('--workers', type=int, default=2)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        parser.error('CUDA khong kha dung trong Python hien tai.')
    nms(torch.zeros((1, 4), device='cuda'), torch.ones(1, device='cuda'), 0.45)
    torch.set_float32_matmul_precision('high')
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'train').exists():
        parser.error('Run da ton tai. Chon --output rieng de giu ket qua cu.')
    data, manifests = prepare_dataset(args.dataset.resolve(), output)

    model = YOLO('yolov8n.yaml')
    model.train(data=str(data), epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
                workers=args.workers, device=0, pretrained=False, amp=False,
                project=str(output), name='train', plots=False, save=True, seed=42,
                fliplr=0.0, degrees=5.0, close_mosaic=min(5, args.epochs // 4))
    checkpoint = output / 'train' / 'weights' / 'best.pt'
    trained = YOLO(str(checkpoint))
    validation = list(predict_split(trained, manifests['val'], args.imgsz, 0))
    val_rows = [measure(validation, i / 100) for i in range(5, 99, 2)]
    selected = max(val_rows, key=lambda row: (row['f1'], row['precision']))
    result = measure(predict_split(trained, manifests['test'], args.imgsz, 0), selected['threshold'])
    write_rows(output / 'validation.csv', val_rows)
    write_rows(output / 'test.csv', [result])
    (output / 'detector_config.json').write_text(json.dumps(dict(
        weights='train/weights/best.pt', imgsz=args.imgsz, iou_nms=0.45, max_det=5,
        threshold=selected['threshold'], pretrained=False, epochs=args.epochs,
    ), indent=2) + '\n', encoding='utf-8')
    print('Selected validation:', selected)
    print('Test:', result)


if __name__ == '__main__':
    main()
