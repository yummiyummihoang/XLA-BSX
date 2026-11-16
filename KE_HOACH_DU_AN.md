# Kế hoạch dự án nhận diện biển số xe cho bãi gửi xe

## 1. Phạm vi và sản phẩm cuối kỳ

Ứng dụng nhận một ảnh tại cổng, hiển thị các biển số phát hiện được, đọc biển người dùng chọn và ghi nhận xe vào hoặc ra sau khi người dùng xác nhận. Màn hình lịch sử cho phép tra cứu biển số, giờ vào, giờ ra và các xe đang gửi.

- **Bản demo chính:** ảnh tải lên, giao diện Streamlit, SQLite, thao tác xác nhận hoặc sửa biển số. Nếu có nhiều biển trong ảnh, người dùng chọn đúng một biển cho mỗi giao dịch.
- **Ràng buộc bắt buộc:** toàn bộ pipeline dùng xử lý ảnh cổ điển. Không dùng pretrain, trọng số tải sẵn, fine-tuning hay OCR đóng gói có mô hình học sẵn. OpenCV và NumPy dùng để triển khai các phép xử lý ảnh và so khớp.
- **Phần phát hiện:** Gaussian → Sobel hoặc Canny → phép đóng → tìm và lọc vùng ứng viên theo hình học.
- **Phần đọc biển:** hiệu chỉnh góc, nhị phân hóa và tách ký tự, sau đó so khớp với bộ mẫu tự cắt và gán nhãn từ tập train của dự án.
- **Mở rộng sau demo:** lấy một khung hình từ webcam và đưa vào cùng pipeline.

## 2. Pipeline tổng thể

| Bài giảng trong `Materials` | Áp dụng trong dự án |
|---|---|
| [4. Histograms](../Materials/4.Histograms.pdf) | Thử cân bằng histogram trên crop tối hoặc ít tương phản. |
| [6. Segmentation](../Materials/6.Segmentation.pdf) | Ngưỡng Otsu làm mặc định để tách ký tự khỏi nền, ngưỡng cục bộ để thử với ảnh sáng không đều. |
| [7. Morphological](../Materials/7.Morphological.pdf) | Phép đóng gom biên thành vùng ứng viên. Phép mở nhẹ có thể lọc nhiễu trên mask ký tự. |
| [8. Linear Processing Filtering](../Materials/8.LinearProcessingFiltering.pdf) | Lọc Gaussian trước khi tìm biên. |
| [9. Template Matching](../Materials/9.TemplateMatching.pdf) | So ký tự đã chuẩn hóa với mẫu tự tạo bằng sai số bình phương trung bình (MSE). |
| [11. Edge Detection](../Materials/11.EdgeDetection.pdf) | So sánh Sobel và Canny để tìm vùng biển. Dùng Hough để thử ước lượng cạnh khi cần chỉnh nghiêng. |

```text
OFFLINE · Chuẩn bị dữ liệu, mẫu ký tự và đánh giá
Ảnh + nhãn khung → rà soát trùng ảnh và nhãn → manifest train/val/test
    ├─ Train → cắt và gán nhãn ký tự → thư viện mẫu theo lớp
    ├─ Val   → chỉnh ngưỡng tìm vùng, tách ký tự và so khớp → khóa cấu hình
    └─ Test  → đánh giá phát hiện, đọc ký tự và toàn pipeline → báo cáo

ONLINE · Ghi nhận xe qua cổng
Ảnh → Gaussian → Sobel/Canny → phép đóng → lọc khung → chọn biển
    → cắt và chỉnh nghiêng → Otsu → tách dòng và ký tự → so khớp mẫu
    → chuẩn hóa chuỗi → hiện kết quả cho người dùng
    → người dùng xác nhận/sửa → chọn Xe vào hoặc Xe ra → ghi SQLite
    → hiện kết quả giao dịch và lịch sử
```

### 2.1. Chuẩn bị dữ liệu

1. Dùng [notebook EDA](kham-pha-dataset.ipynb) và [báo cáo EDA](eda_report_bundle/report_vi.md) làm đầu vào. Số liệu gốc gồm 8.259 ảnh và 8.452 hộp, chia thành 6.607 ảnh train, 814 val và 838 test. Nhãn hiện có là tọa độ khung của một lớp. Dùng `license_plate` làm tên lớp trong dự án sau khi đối chiếu mẫu.
2. Kiểm tra bốn nhóm trùng pixel trong [bảng ảnh trùng](eda_report_bundle/pixel_duplicate_groups.csv), đặc biệt `images/train/quandoi64.jpg` và `images/val/quandoi50.jpg`. Đối chiếu nhãn trước khi giữ một bản. Với cặp train/val, ưu tiên giữ bản val nếu nhãn tương đương. Soát thêm ứng viên gần trùng từ `dhash_candidates.csv`. Giữ nguyên bộ ảnh gốc, ghi lý do giữ hoặc loại trong manifest.
3. Xuất `split_manifest.csv` gồm `image_path, label_path, split, group_id, include, reason`. Kiểm tra mọi ảnh được dùng đọc được, mỗi nhãn ghép đúng ảnh và mỗi nhóm trùng đã xác nhận chỉ xuất hiện ở một split. Báo cáo lại số ảnh và hộp sau khi lọc.
4. Gán thử chuỗi biển thật cho 30 vùng biển từ val. Sau đó mở rộng lên khoảng 100–150 vùng biển đọc được, gồm val để chọn cấu hình và test để chấm điểm cuối. Lưu `image_path, plate_id, bbox_xyxy, plate_text, layout, readable, condition` trong `ocr_labels.csv`. Với biển không đọc được bằng mắt, ghi `readable=false` và thống kê riêng. Kiểm tra lại nhãn nhập tay trước khi khóa tập test.
5. Thu thêm khoảng 30–50 ảnh cổng hoặc xe không lộ biển, kiểm tra thủ công để xác nhận ảnh âm tính. Nếu có thể, chụp thêm một nhóm ảnh có biển tại cổng để kiểm tra chênh lệch giữa dataset và môi trường demo. Tách riêng các ảnh dùng để chỉnh cấu hình và các ảnh dùng để chấm.
6. Từ các crop thuộc train, cắt và gán nhãn từng ký tự. Tạo nhiều mẫu cho mỗi lớp quan sát được, ưu tiên đủ chữ số và các chữ cái xuất hiện trong phạm vi demo. Lưu `character, template_path, source_image_path, source_plate_id` để truy được ảnh nguồn. Ghi lớp thiếu mẫu và trả về trạng thái cần kiểm tra nếu điểm so khớp thấp. Mẫu từ val và test bị loại khỏi thư viện mẫu.

### 2.2. Phát hiện vị trí biển

| Nhánh | Cách làm | Đầu ra |
|---|---|---|
| Sobel | Chuyển xám → Gaussian → lấy độ lớn gradient theo trục x để làm nổi các nét đứng → ngưỡng → phép đóng → tìm contour. | Danh sách khung `xyxy` và điểm theo hình học, độ phủ biên. |
| Canny | Chuyển xám → Gaussian → Canny → phép đóng → tìm contour. | Cùng định dạng đầu ra để so sánh trực tiếp với Sobel. |

Lọc ứng viên theo diện tích tương đối, độ chữ nhật và hai khoảng tỷ lệ rộng/cao cho biển một dòng và hai dòng. Gộp ứng viên trùng vùng theo IoU để tránh hiện một biển nhiều lần. Lưu kernel, ngưỡng biên và tham số lọc đã chọn trên val. Quy đổi tọa độ về ảnh gốc nếu có resize, giới hạn tọa độ trong biên và loại hộp rỗng. Chọn nhánh tích hợp bằng kết quả val và thời gian chạy trên máy demo.

### 2.3. Cắt biển và đọc ký tự

1. Cắt theo khung và thử thêm một biên nhỏ để giữ ký tự sát mép. Giữ bản crop gốc làm đối chứng.
2. Ước lượng góc nhẹ từ cạnh dài của vùng biển hoặc các đường thẳng Hough. Giữ crop gốc nếu đường cạnh thiếu nhất quán. Chuyển xám và thử cân bằng histogram trên val trước khi nhị phân hóa.
3. Dùng Otsu để tạo mask ký tự. Thử ngưỡng cục bộ cho ảnh có bóng hoặc chênh sáng, chuẩn hóa mask để nét chữ là tiền cảnh. Tìm một hoặc hai dải dòng theo tổng số pixel tiền cảnh trên từng hàng.
4. Trong mỗi dòng, gán nhãn thành phần liên thông và lọc bằng chiều cao, bề rộng và diện tích tương đối. Đọc dòng trên rồi dòng dưới, trong mỗi dòng sắp ký tự từ trái qua phải. Với ký tự dính hoặc đứt nét, đánh dấu ca lỗi để kiểm tra khi đánh giá phân đoạn.
5. Đưa mỗi ký tự và mẫu về một khung cố định, giữ tỷ lệ và căn giữa. Với mỗi lớp, lấy MSE nhỏ nhất giữa ký tự và các mẫu lớp đó. Dùng val để đặt ngưỡng lỗi tối đa và khoảng cách tối thiểu giữa hai lớp đứng đầu. Nếu chưa đạt, đánh dấu ký tự `?` và yêu cầu kiểm tra thủ công.
6. Trả về `bbox_xyxy, detection_score, text_raw, text_normalized, match_error, status`. Chuẩn hóa chữ hoa và bỏ dấu phân cách, giữ trạng thái nghi vấn đến khi người dùng sửa. Giao diện xử lý ảnh lỗi, không thấy biển, có nhiều biển, đọc thiếu ký tự và đọc được.

### 2.4. Giao dịch tại bãi xe

| Tình huống | Hành vi |
|---|---|
| Có kết quả OCR | Hiện ảnh, khung và ô biển số cho phép sửa. Chỉ ghi dữ liệu sau khi người dùng xác nhận thao tác. |
| Không có khung hoặc OCR rỗng | Hiện trạng thái phù hợp và cho nhập biển bằng tay nếu người dùng xác minh được. |
| Xe vào | Tạo lượt với `entered_at` và `exited_at` để trống. Nếu biển đã có lượt mở, báo lỗi và giữ lượt hiện tại. |
| Xe ra | Tìm lượt mở theo biển đã xác nhận và cập nhật `exited_at`. Nếu không tìm thấy, báo lỗi để người dùng kiểm tra lại biển. |

Bảng `parking_sessions` gồm `id, plate_normalized, entered_at, exited_at`. Dùng unique index có điều kiện trên biển số cho các lượt chưa có giờ ra. Cập nhật lượt ra bằng điều kiện `exited_at IS NULL` và kiểm tra số dòng được cập nhật. Mỗi thao tác ghi là một giao dịch SQLite để chặn lượt vào trùng khi bấm hai lần hoặc có hai yêu cầu đồng thời.

## 3. Đánh giá và nghiệm thu

| Phần | Cách đo | Đầu ra cần lưu |
|---|---|---|
| Phát hiện | Ghép một-một khung dự đoán và nhãn cùng ảnh tại IoU ≥ 0,5. Báo precision, recall và F1 ở ngưỡng đã chọn trên val. | Bảng Sobel và Canny, cùng mẫu bỏ sót và khoanh nhầm. |
| Ảnh âm tính | Đếm số ảnh âm tính bị báo có ít nhất một biển, chia cho số ảnh âm tính đã xác minh. | Số ảnh lỗi, tổng ảnh và ví dụ. |
| Đọc trên khung nhãn | So chuỗi chuẩn hóa với chuỗi thật. Đo tỷ lệ đúng cả biển và tỷ lệ lỗi ký tự. So Otsu với ngưỡng cục bộ trên cùng mẫu val. | Kết quả trước và sau tiền xử lý, có thống kê theo bố cục nếu đủ mẫu. |
| Toàn pipeline | Với mỗi biển có chuỗi thật đọc được, cần có khung ghép đúng tại IoU ≥ 0,5 và chuỗi đúng. Biển bị bỏ sót hoặc OCR rỗng được tính sai. | Số biển đúng trên tổng biển được chấm, kèm lỗi khoanh, lỗi đọc và thời gian xử lý trên máy demo. |
| Ứng dụng | Chạy xe vào hợp lệ, xe ra hợp lệ, xe vào trùng, xe ra chưa có lượt, sửa tay, ảnh không có biển và ảnh nhiều biển. Khởi động lại ứng dụng rồi kiểm tra lịch sử. | Kết quả từng ca, ảnh chụp demo và lịch sử còn lưu sau khởi động lại. |

Chọn cấu hình bằng val rồi mới chấm test. Chốt mục tiêu độ chính xác sau lần chạy thử trên val và ghi cả số mẫu lẫn tỷ lệ khi báo cáo.

## 4. Lịch triển khai theo mốc

| Mốc | Việc thực hiện | Điều kiện qua mốc |
|---|---|---|
| Tuần 1 · Dữ liệu | Rà ảnh trùng, xuất manifest, gán chuỗi cho 30 biển val và cắt mẫu ký tự từ train. | Split được kiểm tra, có bộ mẫu truy về ảnh gốc và bộ crop val để thử. |
| Tuần 2 · Phát hiện | Làm nhánh Sobel và Canny, bổ sung ảnh âm tính, chốt ngưỡng lọc ứng viên trên val. | Khung trả về theo tọa độ ảnh gốc và có bảng so sánh hai nhánh. |
| Tuần 3 · Đọc biển và tích hợp | So các cách nhị phân hóa, tách ký tự, so khớp mẫu MSE, ghép với bước phát hiện rồi khóa cấu hình và chạy test. | Một lệnh nhận ảnh và xuất kết quả. Có bảng phát hiện, đọc ký tự và toàn pipeline. |
| Tuần 4 · Bãi xe và báo cáo | Tích hợp Streamlit và SQLite, kiểm thử giao dịch, chuẩn bị demo và viết báo cáo. | Chạy được từ ảnh tới lịch sử, qua đủ ca nghiệm thu và có hướng dẫn chạy trên máy khác. |

## 5. Cấu trúc triển khai đề xuất

```text
CuoiKy/
├── License Plate Detection Dataset/  # Dữ liệu gốc
├── eda_report_bundle/                # Kết quả EDA hiện có
├── data/                            # Manifest, nhãn biển, bộ mẫu train và ảnh kiểm thử bổ sung
├── configs/                         # Cấu hình dữ liệu và tham số đã chốt
├── scripts/                         # Chuẩn bị dữ liệu, tạo bộ mẫu, đánh giá
├── src/                             # Phát hiện, tiền xử lý, tách ký tự, so khớp và lưu lượt xe
├── outputs/                         # Số liệu, ảnh trung gian và ảnh lỗi
├── app.py                           # Giao diện Streamlit
├── requirements.txt
└── README.md                        # Hướng dẫn cài đặt, chạy và tái lập kết quả
```

**Bắt đầu:** lập manifest và xử lý ảnh trùng, cắt bộ mẫu ký tự từ train rồi gán chuỗi cho 30 biển val. Dùng khung nhãn của 30 biển này để thử Otsu, tách dòng và so khớp mẫu. Lưu ảnh crop, mask, khung ký tự và điểm MSE để phân biệt lỗi tách ký tự với lỗi so khớp.
