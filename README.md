# Nhận diện biển số xe cho bãi gửi xe

Đồ án cuối kỳ môn Xử lý ảnh tại PTIT. Mục tiêu là phát hiện và đọc biển số từ ảnh chụp tại cổng, sau đó ghi nhận lượt xe vào hoặc ra khi người dùng xác nhận.

## Tiến độ

Repo hiện có [kế hoạch triển khai](KE_HOACH_DU_AN.md), [notebook khám phá dữ liệu](kham-pha-dataset.ipynb) và [tóm tắt EDA](eda_report_bundle/report_vi.md). Phần nhận diện và ứng dụng bãi xe đang ở giai đoạn thiết kế.

## Pipeline dự kiến

1. **Phát hiện vùng biển.** Chuyển xám, lọc Gaussian, tìm biên Sobel hoặc Canny, đóng hình thái và lọc vùng theo hình học.
2. **Đọc ký tự.** Cắt biển, hiệu chỉnh góc nếu xác định được, nhị phân hóa Otsu, tách dòng và so khớp ký tự với mẫu tự tạo từ tập train.
3. **Ghi nhận xe.** Cho phép xác nhận hoặc sửa biển đã đọc rồi lưu lượt vào/ra trong SQLite.

Pipeline áp dụng xử lý ảnh cổ điển theo bài giảng và không sử dụng mô hình pretrained. Tập val dùng để chọn tham số và tập test dùng để chấm cuối kỳ.

## Dữ liệu

Tải [Vietnamese License Plate Detection](https://www.kaggle.com/datasets/miahuynh04/vietnamese-license-plate-detection). Để chạy notebook cục bộ, đặt thư mục `License Plate Detection Dataset` cạnh README rồi đặt `ROOT_OVERRIDE` thành đường dẫn tới thư mục này. Khi chạy trên Kaggle, gắn dataset vào notebook và chạy toàn bộ cell. Bộ ảnh gốc và các file EDA dung lượng lớn được giữ ngoài Git. Các liên kết bài giảng trong kế hoạch trỏ tới thư mục `Materials` của workspace môn học.
