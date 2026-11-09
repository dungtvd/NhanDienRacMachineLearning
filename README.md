# Phân loại rác bằng MobileNetV2

Dự án dùng MobileNetV2 để phân loại ảnh rác theo 8 nhóm của MWCD: giấy, vải,
kim loại, nhựa, hữu cơ, thủy tinh, pin/rác nguy hại và rác khác. Ứng dụng
Streamlit hỗ trợ nhận diện ảnh, bổ sung dữ liệu có nhãn xác nhận và học tăng
dần tùy chọn.

## Cài đặt trên Windows

Mở PowerShell tại thư mục dự án:

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

TensorFlow trên Windows chạy CPU; dùng Google Colab có GPU nếu cần huấn luyện
nặng.

## Dataset

Dataset chuẩn là `data/final_merged_dataset/`. Cấu hình
`configs/config_mwcd.yaml` gom các thư mục nguồn thành 8 nhãn và cũng quy định
thứ tự nhãn đầu ra của model. Nếu cần khôi phục dataset, giải nén archive MWCD
sao cho thư mục `final_merged_dataset` nằm trực tiếp bên trong `data/`.

Các bản dataset lịch sử đã được đối chiếu SHA-256 rồi xóa cùng cấu hình và
artifacts riêng của chúng:

- TrashNet: 2.524/2.527 ảnh trùng với MWCD.
- `garbage_classification`: 15.496/15.515 ảnh trùng với MWCD.
- `realwaste-main`: 4.752/4.752 ảnh trùng với MWCD.

Có 22 ảnh không trùng trong ba bản cũ; chúng cũng bị xóa theo lựa chọn chỉ giữ
MWCD. Dataset MWCD gốc vẫn được giữ nguyên.

## Chạy ứng dụng

```powershell
streamlit run app.py
```

Mặc định ứng dụng dùng `models/mwcd_materials/mobilenet_20261029T021239Z.keras`.
Model có 8 đầu ra, đúng thứ tự nhãn trong cấu hình MWCD; file model gốc không
bị sửa. File JSON cùng tên ghi metadata nhãn, accuracy test **91,57%** và
macro F1 **88,89%** trên **6.534 ảnh**. Giao diện hiển thị đúng các metric
được ghi trong JSON, không coi chúng là độ chính xác đảm bảo trên ảnh mới.
Trong đó, **6.534 là tập test**, không phải tổng ảnh huấn luyện. JSON ghi nhận
Thư mục MWCD có **35.168 file ảnh**. Bộ kiểm tra tìm thấy **2.495 ảnh trùng
byte chính xác** và không có ảnh lỗi, nên còn **32.673 ảnh duy nhất**: 19.605
train, 6.534 validation và 6.534 test. Ảnh trùng được loại khi chia tập để
tránh cùng một ảnh xuất hiện ở cả train và test.

Model được lưu bằng Keras 3.13.2, trong khi TensorFlow hiện có thể đi kèm
Keras cũ hơn. Bộ nạp loại bỏ an toàn trường cấu hình `quantization_config`
chỉ khi trường đó có giá trị `null`; weights và model gốc được giữ nguyên.

## Huấn luyện

Các lệnh sau mặc định dùng cấu hình MWCD:

```powershell
python -m src.data.inspect_dataset
python -m src.training.train_cnn
python -m src.training.train_mobilenet
```

MobileNetV2 mặc định tải trọng số ImageNet. Có thể chạy không tải weights
bằng `python -m src.training.train_mobilenet --weights none`. Notebook Colab
trong `notebooks/` cũng đã chuyển sang cấu hình MWCD.

Model đang giữ và JSON tương ứng nằm trong `models/mwcd_materials/`. Các model
thử nghiệm, classifier học tăng dần và báo cáo của model cũ đã được dọn; model
Colab nêu trên là model duy nhất được giữ làm lựa chọn nhận diện.

## Kiểm thử

```powershell
python -m pytest
```

Các điểm số softmax chưa được hiệu chuẩn thành xác suất tin cậy. Kết quả nhận
diện phụ thuộc vào ảnh và dữ liệu dùng để huấn luyện.

Xem [báo cáo chi tiết dự án](docs/project-report.md) để biết kiến trúc, thống
kê dataset, cách chia tập, metric theo lớp và giới hạn đánh giá.
