# 📋 PHÂN CÔNG THUYẾT TRÌNH & LÀM SLIDE - DỰ ÁN ITS
**Nhóm:** 5 thành viên
**Cấu trúc nhóm:**
1.  **Trưởng nhóm IT (Xuân Đạt):** Chuyên về Công nghệ phần mềm, Hệ thống, Kiến trúc, Code vận hành.
2.  **Thành viên IT:** Hỗ trợ hệ thống, Backend, API.
3.  **3 Thành viên DS:** Chuyên về Khoa học dữ liệu, Toán, Mô hình, Đánh giá.

---

## 📅 1. PHÂN CHIA LÀM SLIDE (POWERPOINT)
Mỗi người chịu trách nhiệm làm slide cho phần mình thuyết trình, sau đó Trưởng nhóm sẽ ghép và format lại cho thống nhất.

| STT | Thành viên | Mảng chuyên môn | Nội dung Slide phụ trách |
| :-- | :--- | :--- | :--- |
| **1** | **Thành viên DS 1** | Problem & Data & Preprocessing | **Tổng quan & Xử lý dữ liệu**<br>- Giới thiệu đề tài ITS.<br>- Thực trạng giao thông & Cần thiết của hệ thống.<br>- Bộ dữ liệu (Dataset): Nguồn, Số lượng, Phân bố class.<br>- Tiền xử lý ảnh & Data Augmentation. |
| **2** | **Thành viên DS 2** | Model Arch & Training | **Kiến trúc & Huấn luyện Mô hình**<br>- Giới thiệu CNN & Transfer Learning.<br>- Tại sao chọn MobileNetV2? (So sánh với VGG16/ResNet).<br>- Kiến trúc chi tiết (Base model + Custom head).<br>- Cấu hình Training & Biểu đồ Loss/Accuracy. |
| **3** | **Thành viên DS 3** | Evaluation & Temporal | **Đánh giá & Thuật toán bổ trợ**<br>- Metrics: Accuracy, Precision, Recall, F1-Score.<br>- Confusion Matrix.<br>- **Temporal Confirmation** (Thuật toán xác nhận theo thời gian). |
| **4** | **Thành viên IT** | Backend & API | **Backend & API**<br>- Kiến trúc Backend (FastAPI).<br>- API Endpoints & Request/Response flow.<br>- Tích hợp AI Model vào hệ thống.<br>- Xử lý đa luồng và tối ưu hiệu năng. |
| **5** | **Trưởng nhóm IT (Xuân Đạt)** | System & Demo | **Kiến trúc Hệ thống & Demo**<br>- Sơ đồ kiến trúc tổng thể (Frontend - Backend - AI).<br>- Công nghệ sử dụng (FastAPI, Streamlit, MLflow).<br>- Quy trình triển khai & Vận hành.<br>- **LIVE DEMO**. |

---

## 🎤 2. KỊCH BẢN THUYẾT TRÌNH (SCRIPT)
Thời lượng dự kiến: 15-20 phút.

### **Tổng quan & Xử lý dữ liệu - Thành viên DS 1 (3-4 phút)**
*   "Chào thầy cô và các bạn. Hôm nay nhóm xin trình bày về hệ thống ITS..."
*   Nêu vấn đề: Camera giám sát nhiều nhưng người theo dõi không xuể -> Cần AI cảnh báo tự động.
*   Giới thiệu Dataset: "Chúng em đã thu thập X nghìn ảnh, chia làm 2 nhãn: Bình thường và Sự cố..."
*   **Data Augmentation**: "Vì dữ liệu thực tế rất đa dạng (nắng, mưa, góc quay), nhóm sử dụng kỹ thuật làm giàu dữ liệu..."
*   Show ảnh trước và sau khi xử lý.

### **Kiến trúc & Huấn luyện Mô hình - Thành viên DS 2 (3-4 phút)** [TRỌNG TÂM DATA SCIENCE]
*   Giải thích **Transfer Learning**: "Thay vì train từ đầu, nhóm thừa hưởng tri thức từ ImageNet..."
*   So sánh kỹ thuật: "Nhóm chọn MobileNetV2 vì nó nhẹ, tốc độ nhanh, phù hợp để deploy thực tế hơn là VGG16 quá nặng."
*   Trình bày quá trình train với biểu đồ Loss/Accuracy.
*   Phân tích: "Như thầy cô thấy, Loss giảm dần và hội tụ tại epoch thứ X, không có hiện tượng Overfitting nặng..."

### **Đánh giá & Temporal Confirmation - Thành viên DS 3 (3 phút)** [ĐIỂM SÁNG]
*   Phân tích metrics: Accuracy, Precision, Recall, F1-Score.
*   Giải thích Confusion Matrix.
*   **QUAN TRỌNG:** Trình bày thuật toán **Temporal Confirmation**.
    *   *"Một vấn đề lớn của AI là 'nháy' (flickering) - tức là nhận diện sai trong 1 tích tắc. Để giải quyết, nhóm em (DS team) đã phối hợp với team IT để đưa ra giải thuật Kiểm chứng theo thời gian..."*

### **Backend & API - Thành viên IT (2-3 phút)**
*   Giải thích kiến trúc Backend với **FastAPI**.
*   Trình bày API Endpoints và cách tích hợp AI Model.
*   Flow xử lý: Request -> Preprocessing -> Model Inference -> Response.
*   Các kỹ thuật tối ưu: Async processing, caching, batch processing.

### **Kiến trúc Hệ thống & Demo - Trưởng nhóm IT (Xuân Đạt) (4-5 phút)** [CHỐT HẠ]
*   **Kiến trúc tổng thể:** "Đây không chỉ là một model notebook, mà là một hệ thống hoàn chỉnh."
    *   Trình bày Flow: Camera -> API (FastAPI) -> AI Model -> Dashboard (Streamlit).
*   **Công nghệ & Quy trình triển khai:** FastAPI, Streamlit, MLflow, Docker (nếu có).
*   **Giải thích Code (Nếu bị hỏi):** Sẵn sàng mở VS Code giải thích file `app.py`, `start_api.py`.
*   **LIVE DEMO:**
    *   Chạy `he_thong.bat`.
    *   Upload thử 1 video tai nạn -> Hệ thống cảnh báo.
    *   Show log của API đang chạy ngầm.

---

## ❓ 3. BỘ CÂU HỎI Q&A (DỰ ĐOÁN & PHÂN CÔNG TRẢ LỜI)

### **Nhóm A: Câu hỏi về Mô hình & Dữ liệu (Dành cho 3 bạn DS)**

**Q1: Tại sao độ chính xác (Accuracy) cao nhưng vẫn báo sai?**
*   **Người trả lời:** Thành viên DS 3.
*   **Gợi ý:** "Dạ, vì bộ dữ liệu có thể bị mất cân bằng (Imbalanced). Accuracy không phản ánh hết. Nhóm em quan tâm hơn đến chỉ số **Recall** (để không bỏ sót sự cố) và **Precision** (để giảm báo động giả). Mời thầy xem Confusion Matrix ạ."

**Q2: Làm sao để cải thiện model này tốt hơn nữa?**
*   **Người trả lời:** Thành viên DS 2.
*   **Gợi ý:** "Có 3 cách ạ: 1. Thu thập thêm dữ liệu (đặc biệt là ban đêm/mưa). 2. Dùng Model lớn hơn như EfficientNet (đánh đổi tốc độ). 3. Fine-tune sâu hơn (unfreeze nhiều layer hơn)."

**Q3: Transfer Learning freeze bao nhiêu layer? Tại sao?**
*   **Người trả lời:** Thành viên DS 2.
*   **Gợi ý:** "Nhóm freeze toàn bộ phần base (feature extractor) và chỉ train phần head (classification). Lý do là vì dữ liệu nhóm em chưa đủ lớn để train lại toàn bộ, nếu unfreeze sớm sẽ làm hỏng weights đã học tốt từ ImageNet."

**Q4: Temporal Confirmation hoạt động như thế nào?**
*   **Người trả lời:** Thành viên DS 3.
*   **Gợi ý:** "Dạ, nó giống như việc 'uốn lưỡi 7 lần trước khi nói'. Hệ thống sẽ chờ xem **K frames liên tiếp** (ví dụ 5 frames) đều báo là 'Sự cố' thì mới phát cảnh báo chính thức. Việc này loại bỏ nhiễu do rung lắc camera hoặc vật thể bay qua nhanh."

**Q5: Data Augmentation có ảnh hưởng như thế nào đến kết quả?**
*   **Người trả lời:** Thành viên DS 1.
*   **Gợi ý:** "Data Augmentation giúp model học được các biến thể khác nhau của dữ liệu, tăng tính tổng quát và giảm overfitting. Nhóm em đã thử nghiệm và thấy accuracy tăng X% khi áp dụng augmentation."

### **Nhóm B: Câu hỏi về Hệ thống & Code (Dành cho Team IT)**

**Q6: Tại sao dùng FastAPI mà không dùng Flask/Django?**
*   **Người trả lời:** Thành viên IT.
*   **Gợi ý:** "FastAPI nhanh hơn (Asynchronous), hỗ trợ sẵn Swagger UI (dễ demo và test), và code gọn gàng modern Python (Type hints). Đặc biệt phù hợp cho ML serving vì có thể xử lý nhiều request đồng thời."

**Q7: Hệ thống này có chạy realtime được không?**
*   **Người trả lời:** Trưởng nhóm IT (Xuân Đạt).
*   **Gợi ý:** "Hiện tại trên máy cá nhân đạt ~10-15 FPS. Nếu deploy thực tế, em sẽ dùng thêm **TensorRT** để tối ưu model và chạy trên GPU server hoặc Jetson Nano, khi đó hoàn toàn có thể đạt realtime 30 FPS."

**Q8: Em tổ chức code như thế nào? (Câu hỏi soi code)**
*   **Người trả lời:** Trưởng nhóm IT (Xuân Đạt).
*   **Gợi ý:** "Em tổ chức theo mô hình Modular.
    *   `src/models`: Chứa định nghĩa model.
    *   `src/training`: Logic huấn luyện riêng biệt.
    *   `src/serving`: API để tách biệt việc phục vụ model.
    *   Điều này giúp team DS có thể update model mà không ảnh hưởng code API của team hệ thống."

**Q9: Nếu nhiều camera cùng gửi về thì hệ thống xử lý sao?**
*   **Người trả lời:** Trưởng nhóm IT (Xuân Đạt) hoặc Thành viên IT.
*   **Gợi ý:** "Hiện tại đây là bản Demo Single-stream. Để scale lên, em sẽ cần dùng **Message Queue** (như Kafka/RabbitMQ) để hứng dữ liệu từ camera, sau đó có nhiều Workers chạy model AI để xử lý song song (Horizontal Scaling)."

**Q10: API endpoints được thiết kế như thế nào?**
*   **Người trả lời:** Thành viên IT.
*   **Gợi ý:** "Chúng em thiết kế RESTful API với các endpoints chính: `/predict` cho dự đoán đơn lẻ, `/predict/batch` cho batch processing, `/health` cho health check. Mỗi endpoint có validation đầu vào và error handling đầy đủ."

---

## 📝 4. CHECKLIST CHUẨN BỊ
*   **Thành viên DS (3 bạn):**
    *   [ ] Nắm chắc lý thuyết CNN, Transfer Learning, Metrics.
    *   [ ] Thuộc kịch bản phần mình.
    *   [ ] Chuẩn bị các biểu đồ, hình ảnh minh họa.
*   **Thành viên IT:**
    *   [ ] Review code API và Backend.
    *   [ ] Nắm vững kiến trúc hệ thống.
    *   [ ] Chuẩn bị giải thích về endpoints và tối ưu hóa.
*   **Trưởng nhóm IT (Xuân Đạt):**
    *   [ ] Kiểm tra môi trường Demo (chạy thử trước 30p).
    *   [ ] Chuẩn bị sẵn các file video test "đẹp" (dễ nhận diện).
    *   [ ] Review toàn bộ code để sẵn sàng mở file khi thầy hỏi.
    *   [ ] Ghép và format lại tất cả slides cho thống nhất.

*Chúc nhóm mình A+!* 🚀

