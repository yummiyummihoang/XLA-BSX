# Nhận diện biển số xe cho bãi gửi xe

Đồ án cuối kỳ môn Xử lý ảnh tại PTIT. Mục tiêu là phát hiện và đọc biển số từ ảnh chụp tại cổng, sau đó ghi nhận lượt xe vào hoặc ra khi người dùng xác nhận.

## Tiến độ

Repo hiện có [kế hoạch triển khai](KE_HOACH_DU_AN.md), [notebook khám phá dữ liệu](kham-pha-dataset.ipynb), [tóm tắt EDA](eda_report_bundle/report_vi.md) và cấu trúc mã nguồn ban đầu. Các mô-đun nhận diện và ghi lượt xe đang chờ triển khai.

## Cấu trúc và chạy giao diện

```text
app.py       Giao diện Streamlit ban đầu
src/         Phát hiện, tiền xử lý, tách ký tự, so khớp, pipeline và lưu lượt xe
data/        Manifest, nhãn ký tự và mô tả dữ liệu cục bộ
configs/     Tham số xử lý ảnh và cấu hình đánh giá
scripts/     Chuẩn bị dữ liệu, tạo mẫu và đánh giá
outputs/     Kết quả cục bộ, được tạo khi chạy các bước xử lý
```

```powershell
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

## Pipeline dự kiến

1. **Phát hiện vùng biển.** Chuyển xám, lọc Gaussian, tìm biên Sobel, đóng hình thái và lọc vùng theo hình học. Dùng HOG và SVM tuyến tính huấn luyện từ đầu để loại vùng nhiễu.
2. **Đọc ký tự.** Cắt biển, hiệu chỉnh góc nếu xác định được, nhị phân hóa Otsu, tách dòng và so khớp ký tự với mẫu tự tạo từ tập train.
3. **Ghi nhận xe.** Cho phép xác nhận hoặc sửa biển đã đọc rồi lưu lượt vào/ra trong SQLite.

Pipeline áp dụng xử lý ảnh cổ điển theo bài giảng và không sử dụng mô hình pretrained. Tập val dùng để chọn tham số và tập test dùng để chấm cuối kỳ.

## Huấn luyện bộ phát hiện

Mở [optimize-svm-detector.ipynb](optimize-svm-detector.ipynb) từ gốc repo với kernel có `numpy`, `opencv-python` và `scikit-learn`, sau đó chạy toàn bộ cell. Notebook chạy trên CPU và học từ ảnh cùng nhãn khung sẵn có. Có thể đặt `SMOKE_RUN = True` để thử nhanh trên một mẫu nhỏ.

Notebook kết hợp Sobel, Canny và black-hat hình thái với nhiều kích thước kernel. SVM chấm điểm ứng viên trước khi khử khung trùng. Kết quả lưu riêng trong `outputs/svm_optimized/`, gồm cấu hình, trọng số và các bảng `validation.csv`, `test.csv`. Giữ `detector_config.json` và `svm_weights.npz` cùng thư mục để lớp `src.detector.PlateDetector` nạp mô hình.

[train-svm-detector.ipynb](train-svm-detector.ipynb) tái lập phiên bản Sobel + HOG + SVM ban đầu. Nếu đã chạy notebook này, `test.csv` của bản tối ưu sẽ có thêm kết quả đối chiếu trên cùng ảnh.

Trên 837 ảnh test tại IoU ≥ 0,5, bản tối ưu đạt precision 82,15%, recall 64,03% và F1 71,97%. Bản Sobel + HOG + SVM ban đầu đạt lần lượt 87,18%, 44,10% và 58,57%. Các số này đánh giá vị trí biển số.

Chạy [refine-svm-detector.ipynb](refine-svm-detector.ipynb) sau notebook tối ưu để huấn luyện hồi quy vị trí và kích thước từ nhãn `train`. Mô hình lưu trong `outputs/svm_refined/` và đạt precision 88,72%, recall 69,58% và F1 77,99% trên cùng tập test. `PlateDetector` nạp mô hình từ `outputs/svm_refined/detector_config.json`.

Để thử YOLOv8n trên GPU NVIDIA, dùng môi trường có PyTorch CUDA, `ultralytics==8.3.235` và chạy từ gốc repo. Script khởi tạo từ YAML, huấn luyện từ trọng số ngẫu nhiên và chấm trên cùng các tập với SVM.

```powershell
python -m scripts.train_yolo --epochs 20
```

Kết quả ghi vào `outputs/yolo_scratch/`. Điều chỉnh `--batch` nếu thiếu bộ nhớ GPU. Dùng `--output` khác cho lần chạy mới.

Lượt thử `--epochs 12 --output outputs/yolo_gpu_trial` chạy trên RTX 5060 Laptop 8 GB, dùng ảnh 640 px và batch 12. Ngưỡng confidence 0,43 được chọn trên val. Trên 837 ảnh test tại IoU ≥ 0,5, lượt này đạt precision 99,41%, recall 98,58% và F1 98,99%. Checkpoint dùng để dự đoán nằm ở `outputs/yolo_gpu_trial/train/weights/best.pt`.

## Dữ liệu

Tải [Vietnamese License Plate Detection](https://www.kaggle.com/datasets/miahuynh04/vietnamese-license-plate-detection). Để chạy notebook cục bộ, đặt thư mục `License Plate Detection Dataset` cạnh README rồi đặt `ROOT_OVERRIDE` thành đường dẫn tới thư mục này. Khi chạy trên Kaggle, gắn dataset vào notebook và chạy toàn bộ cell. Bộ ảnh gốc và các file EDA dung lượng lớn được giữ ngoài Git. Các liên kết bài giảng trong kế hoạch trỏ tới thư mục `Materials` của workspace môn học.
