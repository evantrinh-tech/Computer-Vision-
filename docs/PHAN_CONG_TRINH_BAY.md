# 📋 PHÂN CÔNG THUYẾT TRÌNH & LÀM SLIDE - DỰ ÁN ITS
**Nhóm:** 6 thành viên
**Cấu trúc nhóm:**
1.  **Trưởng nhóm (Bạn - IT):** Chuyên về Công nghệ phần mềm, Hệ thống, Kiến trúc, Code vận hành.
2.  **5 Thành viên (DS 1-5):** Chuyên về Khoa học dữ liệu, Toán, Mô hình, Đánh giá.

---

## 📅 1. PHÂN CHIA LÀM SLIDE (POWERPOINT)
Mỗi người chịu trách nhiệm làm slide cho phần mình thuyết trình, sau đó Trưởng nhóm sẽ ghép và format lại cho thống nhất.

| STT | Thành viên | Mảng chuyên môn | Nội dung Slide phụ trách |
| :-- | :--- | :--- | :--- |
| **1** | **Thành viên DS 1** | Problem & Data | **Tổng quan & Dữ liệu**<br>- Giới thiệu đề tài ITS.<br>- Thực trạng giao thông & Cần thiết của hệ thống.<br>- Bộ dữ liệu (Dataset): Nguồn, Số lượng, Phân bố class. |
| **2** | **Thành viên DS 2** | Preprocessing | **Xử lý dữ liệu (Data Processing)**<br>- Các bước tiền xử lý ảnh (Resize, Normalization).<br>- Kỹ thuật Data Augmentation (Xoay, Lật, Chỉnh sáng) - *Giải thích tại sao cần*. |
| **3** | **Thành viên DS 3** | Model Arch | **Kiến trúc Mô hình (Modeling)**<br>- Giới thiệu CNN & Transfer Learning.<br>- Tại sao chọn MobileNetV2? (So sánh với VGG16/ResNet).<br>- Kiến trúc chi tiết (Base model + Custom head). |
| **4** | **Thành viên DS 4** | Training | **Huấn luyện (Training Process)**<br>- Cấu hình Training (Epochs, Batch size, LR).<br>- Loss function & Optimizer (Adam/CrossEntropy).<br>- Biểu đồ Loss/Accuracy trong quá trình train. |
| **5** | **Thành viên DS 5** | Evaluation | **Đánh giá & Thuật toán bổ trợ**<br>- Metrics: Accuracy, Precision, Recall, F1-Score (Giải thích ý nghĩa).<br>- Confusion Matrix.<br>- **Temporal Confirmation** (Thuật toán xác nhận theo thời gian). |
| **6** | **Trưởng nhóm (IT)** | System & Demo | **Kiến trúc Hệ thống & Demo**<br>- Sơ đồ kiến trúc tổng thể (Frontend - Backend - AI).<br>- Công nghệ sử dụng (FastAPI, Streamlit, MLflow).<br>- **LIVE DEMO**. |

---

## 🎤 2. KỊCH BẢN THUYẾT TRÌNH (SCRIPT)
Thời lượng dự kiến: 15-20 phút.

### **Mở đầu - Thành viên DS 1 (2 phút)**
*   "Chào thầy cô và các bạn. Hôm nay nhóm xin trình bày về hệ thống ITS..."
*   Nêu vấn đề: Camera giám sát nhiều nhưng người theo dõi không xuể -> Cần AI cảnh báo tự động.
*   Giới thiệu sơ lược Dataset: "Chúng em đã thu thập X nghìn ảnh, chia làm 2 nhãn: Bình thường và Sự cố..."

### **Xử lý dữ liệu - Thành viên DS 2 (2 phút)**
*   Nhấn mạnh vào **Data Augmentation**: "Vì dữ liệu thực tế rất đa dạng (nắng, mưa, góc quay), nhóm sử dụng kỹ thuật làm giàu dữ liệu..."
*   Show ảnh trước và sau khi xử lý để thấy sự khác biệt.

### **Mô hình AI - Thành viên DS 3 (3 phút)** [TRỌNG TÂM DATA SCIENCE]
*   Giải thích **Transfer Learning**: "Thay vì train từ đầu, nhóm thừa hưởng tri thứ từ ImageNet..."
*   So sánh kỹ thuật: "Nhóm chọn MobileNetV2 vì nó nhẹ, tốc độ nhanh, phù hợp để deploy thực tế hơn là VGG16 quá nặng."

### **Huấn luyện - Thành viên DS 4 (2 phút)**
*   Trình bày quá trình train.
*   Phân tích biểu đồ: "Như thầy cô thấy, Loss giảm dần và hội tụ tại epoch thứ X, không có hiện tượng Overfitting nặng..."

### **Đánh giá & Giải thuật - Thành viên DS 5 (3 phút)** [ĐIỂM SÁNG]
*   Phân tích sai số: "Model thỉnh thoảng nhầm lẫn ở các trường hợp..."
*   **QUAN TRỌNG:** Trình bày thuật toán **Temporal Confirmation**.
    *   *"Một vấn đề lớn của AI là 'nháy' (flickering) - tức là nhận diện sai trong 1 tích tắc. Để giải quyết, nhóm em (DS team) đã phối hợp với team IT để đưa ra giải thuật Kiểm chứng theo thời gian..."*

### **Hệ thống & Demo - Trưởng nhóm IT (4-5 phút)** [CHỐT HẠ]
*   **Kiến trúc:** "Đây không chỉ là một model notebook, mà là một hệ thống hoàn chỉnh."
    *   Trình bày Flow: Camera -> API (FastAPI) -> AI Model -> Dashboard (Streamlit).
*   **Giải thích Code (Nếu bị hỏi):** Sẵn sàng mở VS Code giải thích file `app.py`, `start_api.py`.
*   **LIVE DEMO:**
    *   Chạy `he_thong.bat`.
    *   Upload thử 1 video tai nạn -> Hệ thống cảnh báo.
    *   Show log của API đang chạy ngầm.

---

## ❓ 3. BỘ CÂU HỎI Q&A (DỰ ĐOÁN & PHÂN CÔNG TRẢ LỜI)

### **Nhóm A: Câu hỏi về Mô hình & Dữ liệu (Dành cho 5 bạn DS)**

**Q1: Tại sao độ chính xác (Accuracy) cao nhưng vẫn báo sai?**
*   **Người trả lời:** Thành viên DS 5.
*   **Gợi ý:** "Dạ, vì bộ dữ liệu có thể bị mất cân bằng (Imbalanced). Accuracy không phản ánh hết. Nhóm em quan tâm hơn đến chỉ số **Recall** (để không bỏ sót sự cố) và **Precision** (để giảm báo động giả). Mời thầy xem Confusion Matrix ạ."

**Q2: Làm sao để cải thiện model này tốt hơn nữa?**
*   **Người trả lời:** Thành viên DS 3 hoặc 4.
*   **Gợi ý:** "Có 3 cách ạ: 1. Thu thập thêm dữ liệu (đặc biệt là ban đêm/mưa). 2. Dùng Model lớn hơn như EfficientNet (đánh đổi tốc độ). 3. Fine-tune sâu hơn (unfreeze nhiều layer hơn)."

**Q3: Transfer Learning freeze bao nhiêu layer? Tại sao?**
*   **Người trả lời:** Thành viên DS 3.
*   **Gợi ý:** "Nhóm freeze toàn bộ phần base (feature extractor) và chỉ train phần head (classification). Lý do là vì dữ liệu nhóm em chưa đủ lớn để train lại toàn bộ, nếu unfreeze sớm sẽ làm hỏng weights đã học tốt từ ImageNet."

**Q4: Temporal Confirmation hoạt động như thế nào?**
*   **Người trả lời:** Thành viên DS 5 (hoặc Trưởng nhóm đỡ nếu bí).
*   **Gợi ý:** "Dạ, nó giống như việc 'uốn lưỡi 7 lần trước khi nói'. Hệ thống sẽ chờ xem **K frames liên tiếp** (ví dụ 5 frames) đều báo là 'Sự cố' thì mới phát cảnh báo chính thức. Việc này loại bỏ nhiễu do rung lắc camera hoặc vật thể bay qua nhanh."

### **Nhóm B: Câu hỏi về Hệ thống & Code (Dành cho Trưởng nhóm IT)**

**Q5: Tại sao dùng FastAPI mà không dùng Flask/Django?**
*   **Trả lời:** Nhanh hơn (Asynchronous), hỗ trợ sẵn Swagger UI (dễ demo và test), và code gọn gàng modern Python (Type hints).

**Q6: Hệ thống này có chạy realtime được không?**
*   **Trả lời:** "Hiện tại trên máy cá nhân đạt ~10-15 FPS. Nếu deploy thực tế, em sẽ dùng thêm **TensorRT** để tối ưu model và chạy trên GPU server hoặc Jetson Nano, khi đó hoàn toàn có thể đạt realtime 30 FPS."

**Q7: Em tổ chức code như thế nào? (Câu hỏi soi code)**
*   **Trả lời:** "Em tổ chức theo mô hình Modular.
    *   `src/models`: Chứa định nghĩa model.
    *   `src/training`: Logic huấn luyện riêng biệt.
    *   `src/serving`: API để tách biệt việc phục vụ model.
    *   Điều này giúp team DS có thể update model mà không ảnh hưởng code API của team hệ thống."

**Q8: Nếu nhiều camera cùng gửi về thì hệ thống xử lý sao?**
*   **Trả lời:** "Hiện tại đây là bản Demo Single-stream. Để scale lên, em sẽ cần dùng **Message Queue** (như Kafka/RabbitMQ) để hứng dữ liệu từ camera, sau đó có nhiều Workers chạy model AI để xử lý song song (Horizontal Scaling)."

---

## 📝 4. CHECKLIST CHUẨN BỊ
*   **Thành viên DS:**
    *   [ ] Nắm chắc lý thuyết CNN, Metrics.
    *   [ ] Thuộc kịch bản phần mình.
*   **Trưởng nhóm IT:**
    *   [ ] Kiểm tra môi trường Demo (chạy thử trước 30p).
    *   [ ] Chuẩn bị sẵn các file video test "đẹp" (dễ nhận diện).
    *   [ ] Review code để sẵn sàng mở file khi thầy hỏi.

*Chúc nhóm mình A+!* 🚀
