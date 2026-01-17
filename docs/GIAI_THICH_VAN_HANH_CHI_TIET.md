# 📖 GIẢI THÍCH VẬN HÀNH CHI TIẾT

Tài liệu này đi sâu vào cách thức hoạt động của từng script chính trong hệ thống, nhằm hỗ trợ việc debug, phát triển thêm tính năng hoặc trả lời các câu hỏi kỹ thuật sâu.

---

## 1. ⚙️ Script: `train_cnn.py` (Huấn luyện Model)

Đây là script chịu trách nhiệm dạy cho AI phân biệt giữa giao thông bình thường và sự cố.

### Luồng hoạt động (Workflow):
1.  **Load Config:** Đọc file `configs/training_config.yaml` để lấy tham số (batch size, epochs...).
2.  **Data Preparation:**
    *   Quét folder `data/images/normal` và `incident`.
    *   Chia dữ liệu thành 3 tập: Train (70%), Validation (15%), Test (15%).
    *   Áp dụng **Data Augmentation** (xoay, lật, chỉnh sáng) cho tập Train để tránh học vẹt.
3.  **Build Model:**
    *   Tải **MobileNetV2** (đã train trên ImageNet).
    *   Đóng băng (Freeze) các lớp feature extraction.
    *   Thêm lớp Dense mới ở cuối để phân loại 2 class.
4.  **Training Loop:**
    *   Chạy vòng lặp theo số Epochs.
    *   Sử dụng **EarlyStopping**: Nếu model không học thêm sau 5 epochs thì tự dừng.
    *   Lưu model tốt nhất vào `models/CNN_model/model.keras`.
5.  **Logging:** Ghi lại lịch sử loss/accuracy lên MLflow.

### Câu hỏi thường gặp về Training:
*   **Q: Tại sao accuracy trên tập train cao mà val thấp?**
    *   **A:** Dấu hiệu Overfitting. Cần tăng cường Data Augmentation hoặc giảm độ phức tạp model.
*   **Q: File `model.keras` nặng bao nhiêu?**
    *   **A:** Khoảng 20-50MB, khá nhẹ nhờ dùng MobileNetV2.

---

## 2. 🔌 Script: `start_api.py` (Backend Server)

Script này biến model thành một Web Service để các ứng dụng khác (Web, Mobile) có thể gọi tới.

### Công nghệ:
*   **FastAPI:** Framework Python hiện đại, hiệu năng cao (Asynchronous).
*   **Uvicorn:** ASGI Server để chạy FastAPI.

### Các Endpoints chính:
*   `POST /predict/image`:
    *   Input: Upload file ảnh.
    *   Process: Resize ảnh -> Chuẩn hóa -> Đưa vào model -> Lấy kết quả.
    *   Output: JSON `{ "prediction": "incident", "confidence": 0.95 }`.
*   `POST /predict/video`:
    *   Input: Upload file video.
    *   Process: Tách frame -> Predict từng frame -> Dùng thuật toán **Temporal Confirmation**.
    *   Output: List các sự cố kèm thời gian bắt đầu/kết thúc.

---

## 3. 🖥️ Script: `app.py` (Frontend Dashboard)

Giao diện người dùng được xây dựng bằng **Streamlit**, giúp tương tác với hệ thống mà không cần code.

### Cấu trúc:
*   **Sidebar:** Menu điều hướng.
*   **Main Area:** Thay đổi nội dung tùy theo menu được chọn.

### Tương tác với Backend:
Khi người dùng bấm "Test Mô hình", Streamlit không tự chạy model trực tiếp (trong mô hình client-server chuẩn) mà sẽ gửi request tới API Server (hoặc gọi module `predictor` nội bộ nếu chạy standalone). Trong dự án này, để đơn giản hóa demo, `app.py` đang import trực tiếp `src.models` để chạy predict.

---

## 4. 🧠 Thuật Toán Bổ Trợ: `temporal_confirmation.py`

Đây là "vũ khí bí mật" giúp hệ thống giảm báo động giả.

### Vấn đề:
Model AI rất nhạy cảm. Chỉ cần một chiếc lá bay qua, hoặc ánh nắng chiếu vào camera làm lóa, model có thể nhận diện nhầm là "Sự cố" trong 1 tích tắc (1 frame).

### Giải pháp:
Không bao giờ tin 1 frame đơn lẻ.
*   Hệ thống duy trì một hàng đợi (Queue) chứa kết quả của K frames gần nhất (ví dụ 5 frames).
*   **Quy tắc:** Cảnh báo chỉ được kích hoạt nếu **cả 5 frames liên tiếp** đều là "Sự cố".

---

## 5. 🛠️ Quy Trình Debug & Sửa Lỗi

### Debug Training
Nếu train bị lỗi, hãy kiểm tra:
1.  Folder `data/images/` có rỗng không?
2.  File ảnh có bị hỏng không? (PIL không mở được).
3.  Learning rate có quá cao làm Loss bị `NaN`?

### Debug API
Nếu API không start được:
1.  Kiểm tra Port 8000 có bị chiếm dụng không.
2.  Kiểm tra logs xem có thiếu thư viện nào không.

---
*Tài liệu hỗ trợ cho đội ngũ vận hành và phát triển.*
