import json
import os
import shutil
from math import isfinite
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import streamlit as st
from PIL import Image, ImageOps, UnidentifiedImageError


ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / 'outputs' / 'yolo_gpu_trial'
SAMPLE = ROOT / 'License Plate Detection Dataset' / 'images' / 'test' / 'boderngoaigiao1.jpg'


@st.cache_resource
def load_detector():
    config_home = ROOT / 'outputs' / 'ultralytics_config'
    config_home.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault('YOLO_CONFIG_DIR', str(config_home))
    import torchvision  # Load native operators before OpenCV on Windows.
    from ultralytics import YOLO

    config = json.loads((MODEL_DIR / 'detector_config.json').read_text(encoding='utf-8'))
    weights = MODEL_DIR / config['weights']
    if not weights.is_file():
        raise FileNotFoundError(weights)
    return YOLO(str(weights)), config


def predict_frame(frame, model, config):
    return model.predict(
        frame, device='cpu', imgsz=config['imgsz'], conf=config['threshold'],
        iou=config['iou_nms'], max_det=config['max_det'], verbose=False,
    )[0]


def show_video(upload, model, config, sample_rate):
    import cv2

    preview = st.empty()
    progress = st.progress(0, text='Đang mở video…')
    processed = 0
    frames_with_plates = 0
    with TemporaryDirectory(prefix='xla_video_') as temporary:
        path = Path(temporary) / ('input' + Path(upload.name).suffix.lower())
        upload.seek(0)
        with path.open('wb') as file:
            shutil.copyfileobj(upload, file)
        capture = cv2.VideoCapture(str(path))
        try:
            if not capture.isOpened():
                raise ValueError('Không mở được video. Hãy thử tệp MP4 mã hóa H.264.')

            fps = capture.get(cv2.CAP_PROP_FPS)
            frame_count = capture.get(cv2.CAP_PROP_FRAME_COUNT)
            has_fps = isfinite(fps) and fps > 0
            has_count = isfinite(frame_count) and frame_count > 0
            step = max(1.0, fps / sample_rate) if sample_rate and has_fps else 1.0
            next_frame = 0.0
            index = 0
            while capture.grab():
                if index >= round(next_frame):
                    ok, frame = capture.retrieve()
                    if not ok:
                        raise ValueError(f'Không đọc được khung hình {index + 1}.')
                    result = predict_frame(frame, model, config)
                    count = len(result.boxes) if result.boxes is not None else 0
                    processed += 1
                    frames_with_plates += int(count > 0)
                    timestamp = f'{index / fps:.1f} giây' if has_fps else f'Khung {index + 1}'
                    preview.image(result.plot(), channels='BGR', caption=f'{timestamp} · {count} biển phát hiện')
                    percent = min(1.0, (index + 1) / frame_count) if has_count else 0.0
                    progress.progress(percent, text=f'Đã xử lý {processed} khung hình')
                    next_frame += step
                index += 1
            if not processed:
                raise ValueError('Video không có khung hình đọc được.')
        finally:
            capture.release()

    progress.progress(1.0, text='Đã xử lý xong video')
    st.success(f'Phát hiện biển ở {frames_with_plates}/{processed} khung hình đã xử lý.')


st.set_page_config(page_title="Nhận diện biển số xe")
st.title("Nhận diện biển số xe cho bãi gửi xe")

mode = st.radio('Nguồn đầu vào', ['Ảnh', 'Video'], horizontal=True)
if mode == 'Video':
    video = st.file_uploader('Video xe ra vào', type=['mp4', 'mov', 'avi', 'mkv', 'webm'])
    sample_rate = st.selectbox('Tần suất phân tích', [3, 5, 10, 0],
                               format_func=lambda rate: f'{rate} khung/giây' if rate else 'Mọi khung hình')
    if video is not None and st.button('Bắt đầu nhận diện', type='primary'):
        try:
            model, config = load_detector()
            show_video(video, model, config, sample_rate)
        except (ValueError, OSError, ImportError, RuntimeError) as error:
            st.error(f'Không xử lý được video. {error}')
    st.stop()


def select_upload():
    st.session_state['sample_selected'] = False


upload = st.file_uploader('Ảnh xe', type=['jpg', 'jpeg', 'png', 'webp'], on_change=select_upload)
if SAMPLE.is_file() and st.button('Thử ảnh mẫu'):
    st.session_state['sample_selected'] = True
source = SAMPLE if st.session_state.get('sample_selected') else upload
if source is None:
    st.stop()

try:
    with Image.open(source) as image:
        photo = ImageOps.exif_transpose(image).convert('RGB')
except (UnidentifiedImageError, OSError):
    st.error('Không đọc được ảnh. Hãy chọn một ảnh JPG, PNG hoặc WebP hợp lệ.')
    st.stop()

try:
    model, config = load_detector()
except (FileNotFoundError, ImportError, OSError, RuntimeError) as error:
    st.error(f'Không nạp được model: {error}')
    st.stop()

with st.spinner('Đang tìm biển số…'):
    result = predict_frame(np.ascontiguousarray(np.asarray(photo)[..., ::-1]), model, config)

if result.boxes is None or len(result.boxes) == 0:
    st.image(photo)
    st.info('Không tìm thấy vùng biển số trong ảnh.')
    st.stop()

st.image(result.plot(), channels='BGR', caption='Vị trí biển số')
for index, (box, score) in enumerate(zip(result.boxes.xyxy.tolist(), result.boxes.conf.tolist()), 1):
    crop = photo.crop(tuple(map(round, box)))
    st.image(crop, caption=f'Biển {index} · Độ tin cậy phát hiện {score:.1%}')
