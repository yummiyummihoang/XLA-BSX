import json
import os
from pathlib import Path

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


st.set_page_config(page_title="Nhận diện biển số xe")
st.title("Nhận diện biển số xe cho bãi gửi xe")


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
    result = model.predict(
        np.ascontiguousarray(np.asarray(photo)[..., ::-1]), device='cpu', imgsz=config['imgsz'],
        conf=config['threshold'], iou=config['iou_nms'], max_det=config['max_det'], verbose=False,
    )[0]

if result.boxes is None or len(result.boxes) == 0:
    st.image(photo)
    st.info('Không tìm thấy vùng biển số trong ảnh.')
    st.stop()

st.image(result.plot(), channels='BGR', caption='Vị trí biển số')
for index, (box, score) in enumerate(zip(result.boxes.xyxy.tolist(), result.boxes.conf.tolist()), 1):
    crop = photo.crop(tuple(map(round, box)))
    st.image(crop, caption=f'Biển {index} · Độ tin cậy phát hiện {score:.1%}')
