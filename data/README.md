# Dữ liệu triển khai

- `split_manifest.csv` lưu đường dẫn ảnh, nhãn khung, split và quyết định lọc trùng.
- `ocr_labels.csv` lưu chuỗi biển đúng để đánh giá trên val và test.
- `templates/` chứa mẫu ký tự tự cắt và gán nhãn từ train. Giữ thông tin ảnh nguồn để kiểm tra rò rỉ giữa các tập.
- `gate_samples/` chứa ảnh kiểm thử bổ sung tại cổng.
