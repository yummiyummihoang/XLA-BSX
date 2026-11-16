# Báo cáo EDA – Vietnamese License Plate Detection
Nguồn: https://www.kaggle.com/datasets/miahuynh04/vietnamese-license-plate-detection
Thời điểm chạy (UTC): 2026-10-12T15:56:41.567901+00:00
- Dataset đọc từ: `/kaggle/input/datasets/miahuynh04/vietnamese-license-plate-detection/License Plate Detection Dataset`.
- Có **8,259 sample ảnh**, 8,259 file nhãn, 8,452 dòng annotation.
- **8,259 ảnh có file nhãn**; **8,259 ảnh ghép duy nhất có ít nhất một box hợp lệ**.
- 8,259 ảnh dương tính đọc được, nhãn ghép duy nhất và không có dòng lỗi.
- 0 ảnh lỗi đọc; 0 ảnh thiếu nhãn; 0 file nhãn rỗng.
- Có 0 dòng nhãn lỗi/khác định dạng; 0 file nhãn mồ côi.
- Các class ID hợp lệ quan sát: [0]; tên có tài liệu: chưa có.
- Trùng pixel: 4 nhóm; 1 nhóm xuất hiện ở nhiều split.
- Chưa xác định số xe máy/ô tô từ annotation hiện có; cần nhãn loại xe hoặc kiểm đếm đã xác minh.
- Chiều rộng 216–4653 px, chiều cao 159–2910 px.
- Kích thước trung vị: rộng 472 px, cao 303 px (hai trung vị riêng).
- Định dạng decode: JPEG: 8258, WEBP: 1.

## Theo split
split,samples,readable,label_file,missing_label,empty_label,invalid_label,unreadable_label,ambiguous_key,has_valid_box,ready_positive,valid_boxes,sample_percent
test,838,838,838,0,0,0,0,0,838,838,849,10.146506841021916
train,6607,6607,6607,0,0,0,0,0,6607,6607,6785,79.99757839932195
val,814,814,814,0,0,0,0,0,814,814,818,9.855914759656132

## Giới hạn và việc cần làm
- Sample là file ảnh; box là dòng annotation. Các số gốc chưa loại annotation/ảnh trùng.
- Nhãn rỗng chưa chắc là ảnh âm tính; thiếu nhãn không chứng minh không có biển số.
- dHash chỉ phát hiện ứng viên collision, chưa dò đầy đủ ảnh gần trùng.
- Xác minh tên lớp và nguồn ảnh; review lỗi nhãn, ảnh nhỏ, ảnh tối và rò rỉ giữa split.
- Không suy loại xe từ hình dạng biển số. Đếm xe cần nhãn loại xe hoặc kiểm đếm trực quan có coverage rõ ràng.