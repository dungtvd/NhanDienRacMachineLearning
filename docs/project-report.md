# Báo cáo chi tiết dự án phân loại rác bằng hình ảnh

**Ngày lập:** 09/11/2026  
**Phiên bản báo cáo:** trạng thái dự án tại thời điểm kiểm tra  
**Model đang sử dụng:** `mobilenet_20261029T021239Z.keras`

## 1. Tóm tắt điều hành

Dự án là ứng dụng phân loại ảnh rác dùng MobileNetV2 và Streamlit. Cấu hình
được giữ lại hiện nay dùng bộ dữ liệu MWCD, gom ảnh nguồn thành tám nhóm vật
liệu. Model MobileNetV2 được lưu từ Google Colab, tích hợp vào ứng dụng cùng
metadata nhãn và số liệu đánh giá.

Các số liệu dataset và đánh giá trong báo cáo được đối chiếu từ dữ liệu đang
có trong workspace, file JSON của model, và lệnh kiểm tra dataset của dự án:

- Dataset MWCD có **35.168 file ảnh**.
- Bộ kiểm tra tìm thấy **2.495 ảnh trùng byte chính xác**, không tìm thấy ảnh
  lỗi; còn **32.673 ảnh duy nhất** sau khi loại bản sao.
- Theo metadata, số ảnh được chia thành **19.605 train**, **6.534 validation**
  và **6.534 test**.
- JSON của model ghi nhận **91,57% accuracy** và **88,89% macro F1** trên
  tập test 6.534 ảnh.
- **6.534 là số ảnh test**, không phải tổng ảnh của dataset hoặc tổng ảnh
  huấn luyện.

Ứng dụng hiển thị metric này như kết quả được lưu trong JSON; đó không phải
cam kết độ chính xác cho từng ảnh mới. Báo cáo này không thực hiện huấn luyện
lại model hoặc đánh giá lại toàn bộ tập test.

## 2. Mục tiêu và phạm vi

### Mục tiêu

- Nhận ảnh rác và dự đoán một trong tám nhóm vật liệu.
- Dùng trực tiếp model MobileNetV2 đã được huấn luyện trên Colab.
- Hiển thị các số liệu đánh giá có sẵn mà không nhầm số lượng test với tổng
  kích thước dataset.
- Cho phép người dùng bổ sung ảnh có nhãn do người dùng xác nhận.
- Cung cấp các công cụ kiểm tra dataset và huấn luyện lại khi có nhu cầu.

### Phạm vi phiên bản hiện tại

- Cấu hình hoạt động: [`configs/config_mwcd.yaml`](../configs/config_mwcd.yaml).
- Dataset chính: `data/final_merged_dataset/`.
- Model nhận diện: `models/mwcd_materials/mobilenet_20261029T021239Z.keras`.
- Metadata và metric: `models/mwcd_materials/mobilenet_20261029T021239Z.json`.
- Giao diện: [`app.py`](../app.py).
- Model CNN và các model thử nghiệm cũ đã được dọn; trong `models/` hiện chỉ
  còn model được chọn cùng JSON tương ứng.

## 3. Chức năng chính

1. **Tổng quan:** hiển thị model, số lượng ảnh dataset, số ảnh duy nhất sau
   khi loại trùng, cách chia train/validation/test và metric trong JSON.
2. **Nhận diện:** nhận ảnh JPEG, PNG hoặc BMP; resize theo kích thước đầu vào
   model; chạy dự đoán và xếp hạng điểm softmax theo lớp.
3. **Bổ sung dữ liệu:** nhận một hoặc nhiều ảnh với nhãn người dùng xác nhận,
   hoặc ZIP có cấu trúc thư mục theo tên lớp. ZIP được đọc trong bộ nhớ thay
   vì giải nén đường dẫn tùy ý ra ổ đĩa.
4. **Báo cáo:** đọc metric từ file kết quả nếu có; nếu thư mục kết quả trống,
   có thể dựng báo cáo và ma trận nhầm lẫn từ `test_metrics` của metadata
   model.
5. **Huấn luyện:** các CLI và notebook Colab hiện mặc định sử dụng cấu hình
   MWCD. Chạy lệnh huấn luyện sẽ tạo artifact mới; không tự chạy khi mở app.

Học tăng dần từng có trong codebase, nhưng trạng thái và artifact classifier
đã bị dọn theo yêu cầu chỉ giữ model MobileNetV2 được chọn. Vì vậy không nên
coi học tăng dần là một classifier đang hoạt động trong bản workspace hiện
tại.

## 4. Dataset và ánh xạ nhãn

Thư mục dữ liệu có **9 thư mục lớp nguồn**. Cấu hình kết hợp chúng thành tám
nhãn đầu ra:

| Nhãn nội bộ | Tên hiển thị | Thư mục nguồn |
|---|---|---|
| `paper` | Giấy | `paper`, `cardboard` |
| `textile` | Vải | `textile` |
| `metal` | Kim loại | `metal` |
| `plastic` | Nhựa | `plastic` |
| `organic` | Hữu cơ | `organic` |
| `glass` | Thủy tinh | `glass` |
| `hazardous` | Pin - rác nguy hại | `battery` |
| `other` | Rác khác | `trash` |

### Số ảnh theo lớp

Các số sau là số ảnh hợp lệ, duy nhất theo nhãn trong metadata model, sau khi
loại ảnh trùng:

| Lớp | Số ảnh |
|---|---:|
| Giấy (`paper`) | 5.649 |
| Vải (`textile`) | 10.960 |
| Kim loại (`metal`) | 2.488 |
| Nhựa (`plastic`) | 3.385 |
| Hữu cơ (`organic`) | 2.531 |
| Thủy tinh (`glass`) | 4.177 |
| Pin - rác nguy hại (`hazardous`) | 1.701 |
| Rác khác (`other`) | 1.782 |
| **Tổng ảnh duy nhất** | **32.673** |

Phân bố lớp không đồng đều: lớp nhiều ảnh nhất là vải (10.960), lớp ít nhất là
pin - rác nguy hại (1.701), tỷ lệ khoảng **6,44:1**. Sự mất cân bằng này cần
được lưu ý khi đọc accuracy tổng thể; macro F1 và metric từng lớp giúp thể
hiện rõ hơn hiệu quả trên lớp ít ảnh.

### Vì sao có 35.168 file nhưng metadata model ghi 32.673 ảnh?

`data/final_merged_dataset/` hiện có 35.168 file ảnh. Dataset loader đọc ảnh,
kiểm tra tính hợp lệ và băm SHA-256 để nhận diện file có nội dung giống hệt.
Ảnh trùng cùng nhãn chỉ giữ một bản; nếu cùng nội dung bị gán vào nhiều nhãn,
các bản sao mơ hồ được loại khỏi tập dùng cho model.

Kết quả kiểm tra trực tiếp:

| Thành phần | Số lượng |
|---|---:|
| File ảnh trong dataset | 35.168 |
| Ảnh trùng chính xác bị loại | 2.495 |
| Ảnh lỗi bị loại | 0 |
| Ảnh duy nhất được giữ | 32.673 |

Việc loại trùng nhằm giảm nguy cơ cùng ảnh xuất hiện ở cả train và test, làm
metric đánh giá cao giả tạo. Kiểm tra hiện tại xác nhận trùng **byte chính
xác**, không khẳng định phát hiện được ảnh gần giống hoặc ảnh đã bị chỉnh
kích thước/nén lại.

Các trường `source_image_files`, `exact_duplicate_images_excluded`,
`invalid_images_excluded` và `unique_images_used` trong JSON hiện tại được bổ
sung tại workspace từ kết quả kiểm tra dataset; đây không phải metric dự đoán
do Colab tự sinh. Con số lớp và kích thước train/validation/test vẫn đối chiếu
với metadata huấn luyện của model.

## 5. Chia tập và cách hiểu số liệu

| Tập | Ảnh | Tỷ lệ xấp xỉ | Vai trò |
|---|---:|---:|---|
| Train | 19.605 | 60% | Cập nhật trọng số model |
| Validation | 6.534 | 20% | Theo dõi quá trình huấn luyện/chọn model |
| Test | 6.534 | 20% | Đánh giá cuối theo metadata |
| **Tổng** | **32.673** | **100%** | Sau loại ảnh trùng |

Các tỷ lệ là xấp xỉ do việc chia được thực hiện riêng theo từng lớp và làm
tròn số ảnh theo lớp. Các metric test phải được đọc cùng số lượng test, không
phải với tổng số ảnh train/validation/test.

## 6. Model và tiền xử lý

- **Kiến trúc:** MobileNetV2 với đầu ra phân loại 8 lớp.
- **Kích thước đầu vào:** 224 × 224 × 3.
- **Đầu vào ảnh:** chuyển RGB, resize về kích thước cấu hình và chuyển sang
  float.
- **Chuẩn hóa:** layer `Rescaling` bên trong model đưa pixel từ `[0, 255]` về
  `[-1, 1]`.
- **Đầu ra:** softmax 8 điểm, theo đúng thứ tự
  `paper`, `textile`, `metal`, `plastic`, `organic`, `glass`, `hazardous`,
  `other`.
- **Artifact:** file `.keras` giữ nguyên; file JSON cùng tên chứa tên lớp,
  số lượng dữ liệu, cấu hình chia tập, lịch sử huấn luyện và metric test.
- **Tương thích Keras:** artifact Colab được lưu bằng Keras 3.13.2. Bộ nạp
  hiện tại tương thích trường `quantization_config: null` mà không ghi sửa
  model gốc.

### Cấu hình huấn luyện ghi trong JSON

| Tham số | Giá trị ghi nhận |
|---|---:|
| Epoch tối đa cấu hình | 30 |
| Patience | 5 |
| Learning rate | 0,001 |
| Fine-tune backbone | Tắt |
| Số vòng trong training history được lưu | 10 |
| Thời gian huấn luyện ghi nhận | khoảng 403,38 giây |

Đây là thông tin được lưu trong JSON. Workspace không có log Colab gốc để
kiểm tra lại trạng thái runtime/GPU hoặc từng bước huấn luyện.

Điểm softmax chỉ là score của model. Chúng chưa được hiệu chuẩn thành xác suất
tin cậy và không đại diện cho mức độ chắc chắn đã được kiểm định.

## 7. Kết quả đánh giá ghi trong JSON

### Chỉ số tổng thể

| Chỉ số | Giá trị |
|---|---:|
| Accuracy | **91,57%** |
| Precision macro | **89,43%** |
| Recall macro | **88,49%** |
| Macro F1 | **88,89%** |
| Số ảnh test | **6.534** |
| Dự đoán đúng suy ra từ confusion matrix | **5.983 / 6.534** |
| Thời gian suy luận trung bình ghi trong JSON | khoảng **3,13 ms/ảnh** |

Accuracy khớp với đường chéo của ma trận nhầm lẫn: 5.983 dự đoán đúng trên
6.534 ảnh, tương đương 91,57%. Thời gian suy luận là số liệu môi trường đánh
giá ghi trong JSON; không thể xem là cam kết tốc độ trên máy Windows khác.

### Metric từng lớp trên tập test

| Lớp | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| Giấy | 93,77% | 90,62% | 92,17% | 1.130 |
| Vải | 97,27% | 99,13% | 98,19% | 2.192 |
| Kim loại | 86,96% | 84,34% | 85,63% | 498 |
| Nhựa | 77,81% | 84,93% | 81,21% | 677 |
| Hữu cơ | 94,25% | 93,87% | 94,06% | 506 |
| Thủy tinh | 89,02% | 88,38% | 88,70% | 835 |
| Pin - rác nguy hại | 95,60% | 95,88% | 95,74% | 340 |
| Rác khác | 80,77% | 70,79% | 75,45% | 356 |

Các lớp có F1 thấp hơn trong báo cáo này là `other` (75,45%) và `plastic`
(81,21%). Đây là điểm cần ưu tiên nếu sau này bổ sung dữ liệu và cải thiện
model.

### Giới hạn khi diễn giải

- Đây là metric **được ghi trong metadata Colab**; lần làm báo cáo này kiểm
  tra tính nhất quán giữa metric và confusion matrix, không chạy lại toàn bộ
  phép đánh giá trên tập test.
- Chất lượng ảnh, cách chụp, vật thể nhiều vật liệu và dữ liệu ngoài MWCD có
  thể làm kết quả thực tế khác.
- Accuracy tổng thể chịu ảnh hưởng phân bố lớp. Macro F1 và metric theo lớp
  cũng nên được xem khi so sánh model.
- Không có ngưỡng score nào được hiệu chuẩn/xác nhận để biến softmax thành
  xác suất tin cậy.

## 8. Kiến trúc phần mềm

| Thành phần | Trách nhiệm |
|---|---|
| `app.py` | Điều hướng Streamlit, trang tổng quan/nhận diện/bổ sung dữ liệu/học tăng dần/báo cáo |
| `src/data/dataset_loader.py` | Tìm thư mục lớp, kiểm tra ảnh, loại ảnh trùng và chia tập |
| `src/data/upload_validation.py` | Kiểm tra ảnh tải lên và ZIP; không giải nén đường dẫn ZIP ra đĩa |
| `src/inference/predict.py` | Đọc/chuyển ảnh và gọi dự đoán Keras |
| `src/inference/model_loader.py` | Nạp model Keras và xử lý metadata cấu hình Keras không tương thích |
| `src/models/mobilenet.py` | Xây MobileNetV2 cho quy trình huấn luyện mới |
| `src/models/cnn.py` | Xây CNN cho thí nghiệm/huấn luyện mới |
| `src/training/common.py` | Nạp cấu hình, cố định seed, pipeline huấn luyện/evaluation |
| `src/evaluation/metrics.py` | Accuracy, precision/recall/F1, confusion matrix và báo cáo |
| `configs/config_mwcd.yaml` | Dataset, thứ tự lớp, chia tập, đường dẫn và giới hạn ứng dụng |
| `notebooks/` | Notebook huấn luyện CNN/MobileNet và continual learning trên Colab |
| `tests/` | Kiểm thử dataset, model, nạp model và kiểm tra upload |

### Bảo vệ dữ liệu tải lên

- Mỗi ảnh riêng lẻ tối đa 10 MB theo cấu hình.
- Chỉ hỗ trợ JPEG, PNG và BMP; kiểm tra nội dung ảnh chứ không chỉ dựa vào
  phần mở rộng.
- Giới hạn kích thước giải mã là 25 megapixel.
- ZIP tải lên tối đa 100 MB, tối đa 500 file và bị giới hạn tổng dung lượng
  giải nén theo cấu hình.
- Từ chối đường dẫn ZIP không an toàn, file không phải ảnh, ảnh hỏng và ảnh
  trùng nội dung nhưng mang nhãn xung đột.
- Ảnh chỉ được đưa vào vùng chờ sau khi người dùng xác nhận nhãn; dự đoán của
  model không tự trở thành nhãn huấn luyện.

## 9. Hướng dẫn cài đặt và chạy

PowerShell trên Windows, tại thư mục dự án:

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

Ứng dụng hiện sử dụng cấu hình MWCD mặc định. Địa chỉ hiện được dùng trong
workspace là `http://localhost:8501`.

Kiểm tra dataset:

```powershell
python -m src.data.inspect_dataset --config configs/config_mwcd.yaml --no-sample-grid
```

Huấn luyện mới theo cấu hình hiện tại:

```powershell
python -m src.training.train_mobilenet --config configs/config_mwcd.yaml
```

Lệnh trên bắt đầu một lần huấn luyện mới; nó không phải thao tác cần thiết để
chạy model Colab đang giữ. Artifact huấn luyện mới sẽ được ghi riêng vào
`models/mwcd_materials/`.

## 10. Kiểm thử và xác minh kỹ thuật

Trong lần kiểm tra hiện tại:

- **25 test tự động thành công**.
- Streamlit AppTest render trang tổng quan không có exception.
- Dashboard hiển thị đúng số liệu dataset/split và metric từ JSON.
- Trang báo cáo dựng một bản ghi cho model còn lại và ma trận nhầm lẫn 8 × 8.
- Model nạp được trong môi trường dự án và tạo đầu ra gồm 8 score.
- Endpoint health của ứng dụng Streamlit đang chạy trả về HTTP 200.

Smoke test nạp model hoặc kiểm tra một ảnh không tương đương với đánh giá chất
lượng trên toàn bộ tập test.

## 11. Trạng thái lưu trữ và dọn dẹp

- Chỉ giữ model `mobilenet_20261029T021239Z.keras` và JSON metadata tương ứng
  trong thư mục model.
- Các checkpoint, model thử nghiệm cũ, trạng thái continual learning và báo
  cáo metrics của model cũ đã được dọn theo yêu cầu chỉ giữ model này.
- Các bộ dữ liệu khác đã được dọn; dataset MWCD chính vẫn còn nguyên.
- Số liệu đánh giá hiện được giữ trong JSON kèm model, không còn phụ thuộc
  vào file báo cáo kết quả riêng trong `results/`.
- `results/` có thể trống cho tới khi chạy kiểm tra dataset có tạo sample grid
  hoặc chạy quy trình đánh giá/huấn luyện mới.

## 12. Hướng cải thiện

1. Giữ nguyên test set làm tập đánh giá cuối, không dùng để chọn model hay
   chỉnh tham số sau khi đã xem kết quả test.
2. Ưu tiên bổ sung ảnh đa dạng, gán nhãn kiểm tra kỹ cho `other` và `plastic`;
   cân nhắc class weight/augmentation khi huấn luyện lại.
3. Bổ sung kiểm tra ảnh gần trùng, ngoài kiểm tra SHA-256 chính xác, trước khi
   tạo split.
4. Đánh giá trên tập ảnh thu thập ngoài MWCD để đo khả năng tổng quát hóa với
   điều kiện chụp thực tế.
5. Nếu muốn hiển thị “độ tin cậy”, hiệu chuẩn score trên validation riêng và
   ghi rõ phương pháp; không dùng trực tiếp softmax như xác suất đã xác thực.
6. Sao lưu model và JSON metadata cùng nhau; JSON cần giữ đúng thứ tự nhãn,
   tiền xử lý, nguồn dataset và metric của đúng phiên bản model.
