# Script triển khai

Các điểm chạy dự kiến gồm `prepare_data.py` để lập manifest, `build_templates.py` để chuẩn hóa các mẫu đã gán nhãn và `evaluate.py` để chấm kết quả phát hiện, đọc ký tự và toàn pipeline.

## Chuẩn bị dữ liệu đọc biển

Chạy từ gốc repo. Script dùng khung nhãn gốc để đánh giá riêng bước đọc.

```powershell
python -m scripts.prepare_ocr --split train --limit 30
python -m scripts.prepare_ocr --split val --limit 30
python -m scripts.prepare_ocr --split val --method adaptive --limit 30
```

1. Mở `outputs/ocr_review/train_otsu/characters.csv` và đối chiếu từng file `glyph_path` với crop gốc. Điền `label` cho ký tự tách đúng. Để trống các vùng nhiễu, ký tự dính hoặc ký tự bị cắt mất nét.
2. Mở `outputs/ocr_review/val_otsu/plates.csv` và nhập chuỗi thật vào `plate_text` theo thứ tự đọc, đánh dấu `readable` cho biển đọc được. Hai bản val dùng cùng bộ ảnh để so sánh và lưu ảnh khoanh ứng viên trong `preview.png`. Ảnh ký tự rời và `characters.csv` chỉ được xuất cho train để tạo bộ mẫu.
3. `src.template_matching.load_train_templates` nạp mẫu đã duyệt từ CSV train. Sau đó có thể dùng `match_rows` và chọn ngưỡng từ tập val. Chỉ báo điểm đọc biển sau khi có chuỗi thật để đối chiếu.
