# GIẢI THÍCH CHI TIẾT CÁC CÔNG NGHỆ & THUẬT NGỮ

## 1️⃣ THUẬT NGỮ CƠ BẢN VỀ AI & MACHINE LEARNING

### Deep Learning (Học Sâu)
- **Là gì**: Một nhánh của AI sử dụng mạng neural nhiều lớp để học từ dữ liệu
- **Ví dụ thực tế**: Giống như bộ não con người học nhận diện khuôn mặt - ban đầu không biết gì, nhưng sau khi nhìn nhiều khuôn mặt, bạn tự động nhận ra được
- **Trong dự án**: Dùng để nhận diện sự cố giao thông từ hình ảnh camera

### Transfer Learning (Học Chuyển Giao)
- **Là gì**: Sử dụng lại kiến thức đã học từ bài toán cũ cho bài toán mới
- **Ví dụ thực tế**: Giống như bạn đã biết chơi piano, khi học organ sẽ dễ hơn vì đã có nền tảng về âm nhạc
- **Trong dự án**: Sử dụng model đã được train trên hàng triệu ảnh (ImageNet) để nhận diện sự cố giao thông, chỉ cần train thêm một chút

### CNN - Convolutional Neural Network (Mạng Neural Tích Chập)
- **Là gì**: Loại mạng neural đặc biệt tốt cho xử lý hình ảnh
- **Cách hoạt động**:
  - **Bước 1**: Nhìn ảnh theo từng phần nhỏ (như quét mắt)
  - **Bước 2**: Tìm các đặc trưng (cạnh, góc, hình dạng)
  - **Bước 3**: Kết hợp các đặc trưng để đưa ra kết luận
- **Ví dụ**: Giống như bạn nhìn một con mèo - trước tiên thấy tai nhọn, râu, đuôi, rồi kết luận "đây là mèo"

### Detection (Phát Hiện)
- **Là gì**: Tìm và xác định vị trí của một đối tượng trong ảnh
- **Trong dự án**: Phát hiện xem có sự cố giao thông hay không

### Segmentation (Phân Vùng)
- **Là gì**: Tô màu chính xác từng pixel thuộc về đối tượng nào
- **Pixel-level**: Chính xác đến từng điểm ảnh
- **Ví dụ**: Giống như tô màu trong tranh - tô đúng từng vùng một

---

## 2️⃣ CÁC MODEL (MÔ HÌNH) ĐƯỢC SỬ DỤNG

### MobileNetV2
- **Là gì**: Một kiến trúc CNN được thiết kế chạy nhanh trên thiết bị di động
- **Đặc điểm**: Nhẹ (48MB), nhanh, phù hợp cho real-time
- **Khi nào dùng**: Khi cần tốc độ, chạy trên máy yếu

### ResNet50
- **Là gì**: CNN với 50 lớp, có kỹ thuật "đường tắt" giúp học tốt hơn
- **Đặc điểm**: Chính xác cao nhưng nặng hơn MobileNet
- **Khi nào dùng**: Khi ưu tiên độ chính xác hơn tốc độ

### VGG16
- **Là gì**: CNN kinh điển với 16 lớp, kiến trúc đơn giản
- **Đặc điểm**: Dễ hiểu, ổn định, nhưng chậm và nặng
- **Khi nào dùng**: Khi cần kiến trúc đơn giản, dễ debug

### U-Net
- **Là gì**: Kiến trúc đặc biệt cho segmentation, hình chữ U
- **Cách hoạt động**:
  - **Nửa trên**: Thu nhỏ ảnh, trích xuất đặc trưng
  - **Nửa dưới**: Phóng to lại, tạo mask phân vùng
- **Trong dự án**: Dùng để tô màu vùng có sự cố

---

## 3️⃣ THUẬT NGỮ VỀ TRAINING (HUẤN LUYỆN)

### Epochs (Kỷ Nguyên)
- **Là gì**: Số lần model nhìn qua TOÀN BỘ tập dữ liệu
- **Ví dụ**: Epochs = 10 nghĩa là model xem hết 10 lần tất cả ảnh training
- **Thực tế**: Epochs càng nhiều, model càng học nhiều (nhưng có thể học quá kỹ - overfitting)

### Batch Size (Kích Thước Lô)
- **Là gì**: Số lượng ảnh xử lý cùng lúc trong 1 lần
- **Ví dụ**: Batch size = 32 nghĩa là mỗi lần xử lý 32 ảnh
- **Trade-off**:
  - **Lớn** → Nhanh nhưng tốn RAM
  - **Nhỏ** → Chậm nhưng ít RAM

### Learning Rate (Tốc Độ Học)
- **Là gì**: Mức độ thay đổi của model mỗi lần học
- **Ví dụ**:
  - Learning rate cao (0.01) → Học nhanh nhưng có thể nhảy quá đích
  - Learning rate thấp (0.0001) → Học chậm nhưng chính xác hơn
- **Tương tự**: Giống như khi đi bộ - bước lớn đi nhanh nhưng dễ vấp, bước nhỏ chậm nhưng an toàn

### Data Augmentation (Tăng Cường Dữ Liệu)
- **Là gì**: Tạo thêm ảnh từ ảnh gốc bằng cách xoay, lật, thay đổi độ sáng...
- **Mục đích**: Tăng số lượng dữ liệu training, giúp model học tổng quát hơn
- **Ví dụ**: Từ 1 ảnh xe hơi, tạo ra 10 ảnh bằng cách xoay, lật, làm tối/sáng

### Train/Validation Split
- **Là gì**: Chia dữ liệu thành 2 phần
  - **Train (80%)**: Dữ liệu để model học
  - **Validation (20%)**: Dữ liệu để kiểm tra xem model có học tốt không
- **Tại sao**: Tránh model học "vẹt" - học thuộc lòng data training

### Callbacks (Hàm Gọi Lại)
Các hàm tự động chạy trong quá trình training:

#### EarlyStopping
- **Là gì**: Dừng training sớm nếu không còn cải thiện
- **Ví dụ**: Nếu 5 epochs liên tiếp không tốt hơn → dừng luôn

#### ModelCheckpoint
- **Là gì**: Lưu model tốt nhất trong quá trình training
- **Ví dụ**: Lưu lại model ở epoch tốt nhất (accuracy cao nhất)

#### ReduceLROnPlateau
- **Là gì**: Tự động giảm learning rate khi model không cải thiện
- **Ví dụ**: Nếu 3 epochs không tốt hơn → giảm learning rate xuống 50%

---

## 4️⃣ METRICS (CHỈ SỐ ĐÁNH GIÁ)

### Accuracy (Độ Chính Xác)
- **Là gì**: Tỷ lệ dự đoán đúng trên tổng số dự đoán
- **Công thức**: (Số dự đoán đúng) / (Tổng số dự đoán) × 100%
- **Ví dụ**: 100 ảnh, đoán đúng 92 → Accuracy = 92%

### Precision (Độ Chính Xác Dương)
- **Là gì**: Trong số dự đoán "có sự cố", bao nhiêu % là đúng
- **Công thức**: (Dự đoán đúng là sự cố) / (Tất cả dự đoán là sự cố)
- **Ví dụ**: Model báo 100 sự cố, thực tế chỉ có 88 sự cố → Precision = 88%

### Recall (Độ Phủ)
- **Là gì**: Trong số sự cố thực tế, model phát hiện được bao nhiêu %
- **Công thức**: (Dự đoán đúng là sự cố) / (Tất cả sự cố thực tế)
- **Ví dụ**: Có 100 sự cố thực tế, model phát hiện 90 → Recall = 90%

### F1-Score
- **Là gì**: Điểm trung bình hài hòa giữa Precision và Recall
- **Công thức**: 2 × (Precision × Recall) / (Precision + Recall)
- **Khi nào dùng**: Khi cần cân bằng giữa Precision và Recall

### Confusion Matrix (Ma Trận Nhầm Lẫn)
Bảng cho biết model dự đoán đúng/sai như thế nào:

```
                    Dự đoán
                Normal   Incident
Thực   Normal     TN        FP
tế     Incident   FN        TP
```

- **TP (True Positive)**: Đoán đúng là sự cố
- **TN (True Negative)**: Đoán đúng là bình thường
- **FP (False Positive)**: Đoán nhầm bình thường thành sự cố (FALSE ALARM)
- **FN (False Negative)**: Bỏ lỡ sự cố thực tế (NGUY HIỂM!)

### False Alarm Rate (Tỷ Lệ Báo Động Giả)
- **Là gì**: Tỷ lệ báo sự cố nhầm khi thực tế không có gì
- **Công thức**: FP / (FP + TN)
- **Trong dự án**: Mục tiêu < 10%

---

## 5️⃣ TEMPORAL CONFIRMATION (XÁC NHẬN THEO THỜI GIAN)

### K-frames Confirmation
- **Là gì**: Xác nhận sự cố qua K frames (khung hình) liên tiếp
- **Ví dụ**: K=3 → chỉ báo sự cố khi 3 frames liên tiếp đều phát hiện sự cố
- **Mục đích**: Giảm false alarm (bóng đổ, ánh sáng lóe... chỉ xuất hiện 1 frame)

### Moving Average (Trung Bình Trượt)
- **Là gì**: Tính trung bình xác suất trong một cửa sổ thời gian
- **Ví dụ**:
  - Frame 1: 70%
  - Frame 2: 80%
  - Frame 3: 90%
  - → Average = 80% → Báo sự cố
- **Lợi ích**: Làm mượt kết quả, ổn định hơn

### Cooldown Period (Thời Gian Hồi)
- **Là gì**: Sau khi báo 1 sự cố, không báo lại trong X giây
- **Ví dụ**: Cooldown = 30s → Sau khi báo sự cố, chờ 30s mới báo sự cố tiếp theo
- **Mục đích**: Tránh spam alerts cho cùng 1 sự cố

---

## 6️⃣ CÔNG NGHỆ BACKEND & API

### FastAPI
- **Là gì**: Framework tạo API (giao diện lập trình) rất nhanh
- **Đặc điểm**:
  - Async (bất đồng bộ) - xử lý nhiều request cùng lúc
  - Tự động tạo docs (Swagger)
  - Type hints - code rõ ràng, ít bug

### REST API
- **Là gì**: Cách để frontend gọi backend qua HTTP
- **Ví dụ**:
  ```
  POST /predict/image
  → Backend: Nhận ảnh, chạy model, trả về kết quả
  ```

### Swagger Documentation
- **Là gì**: Giao diện web tự động để test API
- **Link**: http://localhost:8000/docs
- **Lợi ích**: Không cần code frontend để test API

### Uvicorn
- **Là gì**: Server chạy backend (ASGI server)
- **Vai trò**: Nhận request từ user, chuyển cho FastAPI xử lý

### Pydantic
- **Là gì**: Thư viện kiểm tra và validate dữ liệu
- **Ví dụ**: Đảm bảo confidence score phải từ 0-1, không được âm

---

## 7️⃣ FRONTEND & VISUALIZATION

### Streamlit
- **Là gì**: Framework tạo web app từ Python (không cần HTML/CSS/JS)
- **Đặc điểm**: Cực nhanh, dễ dùng, phù hợp ML engineers
- **Trong dự án**: Tạo dashboard quản lý hệ thống

### Plotly
- **Là gì**: Thư viện vẽ biểu đồ tương tác
- **Ví dụ**: Biểu đồ có thể zoom, hover xem giá trị

### Matplotlib
- **Là gì**: Thư viện vẽ biểu đồ tĩnh (cơ bản nhất)
- **Dùng cho**: Confusion matrix, training curves

---

## 8️⃣ DATABASE & STORAGE

### PostgreSQL
- **Là gì**: Hệ quản trị cơ sở dữ liệu quan hệ (RDBMS)
- **Đặc điểm**: Mạnh mẽ, ổn định, hỗ trợ JSON
- **Trong dự án**: Lưu incidents, predictions, training history

### SQLAlchemy
- **Là gì**: ORM (Object-Relational Mapping)
- **Công dụng**: Viết code Python thay vì SQL
- **Ví dụ**:
  ```python
  # Thay vì SQL: SELECT * FROM incidents WHERE status='confirmed'
  incidents = session.query(Incident).filter_by(status='confirmed').all()
  ```

### Alembic
- **Là gì**: Tool để quản lý database migrations
- **Công dụng**: Thay đổi cấu trúc database một cách an toàn
- **Ví dụ**: Thêm cột mới vào bảng mà không mất dữ liệu cũ

---

## 9️⃣ COMPUTER VISION

### OpenCV
- **Là gì**: Thư viện xử lý hình ảnh và video
- **Công dụng**: Đọc video, resize ảnh, vẽ bounding box...
- **Ví dụ**: Đọc video từ camera RTSP

### RTSP Stream
- **Là gì**: Giao thức truyền video real-time từ camera IP
- **Ví dụ**: rtsp://192.168.1.10:554/stream1
- **Trong dự án**: Kết nối camera giám sát

### Frame
- **Là gì**: Một hình ảnh trong video
- **Ví dụ**: Video 30 FPS = 30 frames mỗi giây

### FPS (Frames Per Second)
- **Là gì**: Số khung hình xử lý được trong 1 giây
- **Trong dự án**:
  - CPU: 3-5 FPS
  - GPU: 20-50 FPS

---

## 🔟 MLOPS & MONITORING

### MLflow
- **Là gì**: Platform quản lý vòng đời Machine Learning
- **Công dụng**:
  - Track experiments (theo dõi thí nghiệm)
  - Log metrics (ghi lại chỉ số)
  - Model registry (kho lưu models)

### Inference
- **Là gì**: Quá trình dùng model đã train để dự đoán
- **Ví dụ**: Đưa ảnh vào model → nhận kết quả

### Latency (Độ Trễ)
- **Là gì**: Thời gian từ khi nhận input đến khi có output
- **Trong dự án**: Mục tiêu < 300ms
- **Ví dụ**: Từ khi upload ảnh đến khi thấy kết quả mất 200ms

### Edge Deployment
- **Là gì**: Chạy AI trên thiết bị biên (Jetson, Coral) thay vì server
- **Lợi ích**: Không cần internet, latency thấp
- **Thiết bị**: NVIDIA Jetson, Google Coral

---

## 1️⃣1️⃣ OPTIMIZATION (TỐI ƯU HÓA)

### TensorRT
- **Là gì**: Thư viện tối ưu hóa model cho GPU NVIDIA
- **Lợi ích**: Tăng tốc 2-10x

### Quantization (Lượng Tử Hóa)
- **Là gì**: Giảm độ chính xác số (float32 → int8)
- **Lợi ích**: Model nhẹ hơn 4x, nhanh hơn 2-4x
- **Trade-off**: Mất chút accuracy (1-2%)

---

## 1️⃣2️⃣ DEPLOYMENT & INFRASTRUCTURE

### Docker
- **Là gì**: Đóng gói ứng dụng và môi trường vào container
- **Lợi ích**: Chạy được mọi nơi, không lo conflict dependencies
- **Ví dụ**: "It works on my machine" → Docker giải quyết

### Kubernetes
- **Là gì**: Quản lý nhiều container, tự động scale
- **Công dụng**: Deploy lên production, auto-restart khi crash

---

## 1️⃣3️⃣ THUẬT NGỮ SYSTEM

### Incident (Sự Cố)
- **Trong dự án**: Tai nạn, xe hỏng, kẹt xe, hành vi bất thường

### Confidence Score (Điểm Tin Cậy)
- **Là gì**: Độ chắc chắn của model về dự đoán
- **Ví dụ**: 0.92 = 92% chắc là sự cố

### Threshold (Ngưỡng)
- **Là gì**: Giá trị ranh giới để quyết định
- **Ví dụ**: Nếu confidence > 0.7 → Báo sự cố

### Pipeline (Đường Ống)
- **Là gì**: Chuỗi các bước xử lý tự động
- **Ví dụ**: Ảnh → Resize → Normalize → Model → Kết quả

---

## 💡 TÓM TẮT QUAN TRỌNG

### Workflow của Hệ thống (Theo Ngôn Ngữ Đơn Giản)

1. **Camera quay** → Ghi hình giao thông
2. **Hệ thống nhận video** → Tách thành từng frame (ảnh)
3. **Tiền xử lý** → Resize ảnh về 224×224, chuẩn hóa màu sắc
4. **Model CNN phân tích** → Nhận diện có sự cố hay không
5. **Temporal Confirmation kiểm tra** → Xác nhận qua nhiều frames
6. **Nếu là sự cố** → Tạo incident, lưu database, gửi alert
7. **Dashboard hiển thị** → Người dùng xem kết quả

### Tại Sao Hệ Thống Này Tốt?

 **Chính xác cao**: 92% accuracy  
 **Nhanh**: < 300ms/ảnh  
 **Giảm báo động giả**: Temporal confirmation  
 **Dễ sử dụng**: Dashboard Streamlit trực quan  
 **Sẵn sàng production**: API, Database, Monitoring đầy đủ

---

*Hy vọng giải thích này giúp bạn hiểu rõ hơn về các công nghệ và thuật ngữ trong dự án! Nếu có thuật ngữ nào còn chưa rõ, hãy hỏi thêm nhé!* 😊
