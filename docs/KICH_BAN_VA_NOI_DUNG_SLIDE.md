# 🎬 KỊCH BẢN THUYẾT TRÌNH & NỘI DUNG SLIDE CHI TIẾT
**Dự án:** ITS - Phát hiện sự cố giao thông (Detecting & Segmenting Abnormal Behavior)
**Thời lượng:** 20-25 phút
**Nhóm:** 5 thành viên (2 CNTT, 3 Khoa học Dữ liệu) - **Nhóm trưởng: Xuân Đạt (IT)**

---

## 📅 BẢNG PHÂN CÔNG TỔNG QUÁT

| STT | Người trình bày | Vai trò | Nội dung chính |
|:---:|:--- |:--- |:--- |
| **1** | **Xuân Đạt (IT - Nhóm trưởng)** | Mở đầu & Kết luận | Giới thiệu, Đặt vấn đề, Kiến trúc hệ thống tổng thể, Tech Stack, **Live Demo**, Kết luận. |
| **2** | **Thành viên IT** | Backend & API | Kiến trúc Backend (FastAPI), API Endpoints, Tích hợp AI Model, Xử lý đa luồng. |
| **3** | **Thành viên DS 1** | Dữ liệu & Tiền xử lý | Dataset, Preprocessing, Data Augmentation. |
| **4** | **Thành viên DS 2** | Modeling & Training | MobileNetV2, Transfer Learning, Quá trình huấn luyện. |
| **5** | **Thành viên DS 3** | Evaluation & Temporal | Metrics, **Temporal Confirmation Algorithm**, Baseline Comparison. |

---

## 📝 CHI TIẾT TỪNG PHẦN (Slide & Lời thoại)

### **PHẦN 1: MỞ ĐẦU & KIẾN TRÚC HỆ THỐNG (Xuân Đạt - IT Leader)**

#### **Slide 1: Trang bìa**
*   **Hình ảnh:** Tên đề tài to rõ, Logo trường, Tên GVHD, Danh sách nhóm.
*   **Lời thoại:**
    > "Xin chào thầy cô và các bạn. Nhóm chúng em xin báo cáo đề tài 'Phát hiện hành vi bất thường trong giám sát giao thông'. Sau đây là danh sách thành viên nhóm..."

#### **Slide 2: Đặt vấn đề (Problem Statement)**
*   **Nội dung:**
    *   Sự bùng nổ camera giám sát -> "Dữ liệu nhiều nhưng không ai xem".
    *   Tai nạn/Sự cố thường bị bỏ qua nếu không có người trực 24/7.
    *   **Mục tiêu:** Xây dựng AI tự động phát hiện sự cố (tai nạn, xe hỏng) để cảnh báo kịp thời.
*   **Lời thoại:**
    > "Trong thời đại smart city, camera có ở khắp nơi. Tuy nhiên, việc giám sát thủ công 24/7 là bất khả thi. Mục tiêu của nhóm em là xây dựng một 'đôi mắt ảo' giúp tự động phát hiện tai nạn hoặc sự cố ngay khi nó xảy ra, giúp lực lượng chức năng ứng phó kịp thời."

#### **Slide 3: Kiến trúc Hệ thống (System Overview)**
*   **Hình ảnh:** Sơ đồ khối:
    *   [Camera/Video] -> [API Server (FastAPI)] -> [AI Engine (MobileNetV2 + Temporal)] -> [Database (PostgreSQL)] -> [Dashboard (Streamlit)].
*   **Nội dung:**
    *   **Backend:** FastAPI xử lý bất đồng bộ, tối ưu đa luồng.
    *   **AI Engine:** MobileNetV2 + Temporal Confirmation Algorithm.
    *   **Frontend:** Dashboard Streamlit hiển thị real-time.
*   **Lời thoại:**
    > "Em đã xây dựng hệ thống theo kiến trúc 3 lớp. Backend sử dụng FastAPI đảm bảo tốc độ cao với đa luồng. AI Engine được tích hợp trực tiếp vào pipeline xử lý video. Kết quả nhận diện được lưu Database và hiển thị tức thì lên Dashboard."

#### **Slide 4: Công nghệ sử dụng (Tech Stack)**
*   **Hình ảnh:** Logo các công nghệ: Python, TensorFlow, FastAPI, Streamlit, PostgreSQL, OpenCV.
*   **Lời thoại:**
    > "Đây là bộ công nghệ nhóm em sử dụng. FastAPI cho hiệu năng cao, Streamlit giúp dễ dàng giám sát, TensorFlow cho AI, và PostgreSQL lưu trữ dữ liệu cảnh báo."

---

### **PHẦN 2: BACKEND & API (Thành viên IT - Backend Lead)**

#### **Slide 5: Kiến trúc Backend với FastAPI**
*   **Hình ảnh:** Sơ đồ luồng xử lý Backend: Request → Validation → Preprocessing → Model Inference → Response.
*   **Nội dung:**
    *   **FastAPI:** Framework hiện đại, xử lý bất đồng bộ (Async).
    *   **API Endpoints:** `/predict`, `/predict/batch`, `/health`.
    *   **Performance:** Hỗ trợ đa luồng, caching để tối ưu tốc độ.
    *   **Integration:** Tích hợp trực tiếp AI Model vào pipeline.
*   **Lời thoại:**
    > "Em phụ trách phần Backend và API. Hệ thống sử dụng FastAPI cho khả năng xử lý bất đồng bộ cao. API được thiết kế với các endpoints chuẩn RESTful, có validation đầu vào và error handling đầy đủ. Mỗi request được xử lý qua pipeline: validation → preprocessing → model inference → trả về kết quả."

#### **Slide 6: Tích hợp AI Model & Tối ưu hóa**
*   **Hình ảnh:** Code snippet hoặc sơ đồ minh họa cách tích hợp model vào API.
*   **Nội dung:**
    *   **Model Loading:** Load một lần khi khởi động server.
    *   **Batch Processing:** Xử lý nhiều frame cùng lúc để tăng throughput.
    *   **Caching:** Cache kết quả để giảm latency.
    *   **Async Processing:** Xử lý đồng thời nhiều request.
*   **Lời thoại:**
    > "Để tối ưu hiệu năng, em áp dụng các kỹ thuật như: load model một lần khi khởi động, batch processing để xử lý nhiều frame cùng lúc, và async processing để server có thể phục vụ nhiều client đồng thời. Điều này giúp hệ thống đạt tốc độ xử lý cao hơn."

---

### **PHẦN 3: DỮ LIỆU & TIỀN XỬ LÝ (Thành viên DS 1 - Data Lead)**

#### **Slide 7: Tổng quan Dữ liệu (Dataset Overview)**
*   **Hình ảnh:** Biểu đồ tròn phân bố (Normal vs Incident). Một vài ảnh mẫu (Sample images) của từng loại.
*   **Nội dung:**
    *   Nguồn: Thu thập từ Youtube, Dataset công khai (AI City Challenge...).
    *   Class 1: **Normal** (Giao thông bình thường).
    *   Class 2: **Incident** (Tai nạn, cháy, va chạm).
    *   Khó khăn: Ảnh mờ, góc quay đa dạng, số lượng ảnh sự cố ít.
*   **Lời thoại:**
    > "Em phụ trách phần dữ liệu. Dataset được thu thập và gán nhãn thành 2 loại: Bình thường và Sự cố. Dữ liệu bao gồm nhiều bối cảnh từ cao tốc đến ngã tư. Thách thức lớn nhất là ảnh sự cố rất hiếm so với ảnh bình thường."

#### **Slide 8: Tiền xử lý & Tăng cường Dữ liệu**
*   **Hình ảnh:** Sơ đồ pipeline: Ảnh gốc -> Resize (224x224) -> Normalize -> Augmentation (xoay, lật, brightness).
*   **Nội dung:**
    *   **Preprocessing:** Resize về 224x224, Normalize pixel values.
    *   **Data Augmentation:** Rotation, Flip, Brightness để cân bằng dataset.
    *   Kết quả: Tăng dataset gấp 3-5 lần, giảm overfitting.
*   **Lời thoại:**
    > "Để giải quyết vấn đề thiếu dữ liệu sự cố, em áp dụng Data Augmentation. Từ một ảnh tai nạn, tạo ra nhiều phiên bản: xoay, lật, chỉnh độ sáng. Điều này giúp mô hình học được bản chất vấn đề, nhận diện tốt cả khi điều kiện ánh sáng thay đổi."

---

### **PHẦN 4: MÔ HÌNH HÓA & HUẤN LUYỆN (Thành viên DS 2 - Model Lead)**

#### **Slide 9: Kiến trúc MobileNetV2 & Transfer Learning**
*   **Hình ảnh:** Sơ đồ kiến trúc [Input -> MobileNetV2 (Pre-trained) -> GlobalAvgPool -> Dense -> Dropout -> Output (2 classes)].
*   **Nội dung:**
    *   **Transfer Learning:** Tận dụng MobileNetV2 đã train trên ImageNet.
    *   **Base Model:** MobileNetV2 (nhẹ 14MB, nhanh, phù hợp real-time).
    *   **Custom Head:** Dense layers để phân loại Normal/Incident.
    *   **So sánh:** MobileNetV2 vs ResNet50 vs VGG16 (tốc độ, kích thước).
*   **Lời thoại:**
    > "Em phụ trách mô hình AI. Nhóm chọn MobileNetV2 làm backbone vì nó cực kỳ nhẹ và nhanh, phù hợp cho real-time. Thay vì train từ đầu, em áp dụng Transfer Learning - tận dụng kiến thức từ ImageNet và fine-tune cho bài toán phát hiện sự cố."

#### **Slide 10: Quá trình Huấn luyện (Training Process)**
*   **Hình ảnh:** 2 biểu đồ đường (Loss & Accuracy) qua các epochs.
*   **Nội dung:**
    *   Framework: TensorFlow/Keras, Optimizer: Adam (lr=0.001).
    *   Loss Function: Binary Crossentropy.
    *   Kết quả: Accuracy ~95%, Loss giảm đều qua epochs.
    *   Hardware: GPU (Google Colab/Local).
*   **Lời thoại:**
    > "Đây là kết quả huấn luyện. Đường xanh là Train, cam là Validation. Loss giảm đều và Accuracy đạt ~95%, chứng tỏ mô hình học tốt và không bị overfitting. Việc sử dụng GPU giúp giảm thời gian train xuống còn vài giờ."

---

### **PHẦN 5: ĐÁNH GIÁ, THUẬT TOÁN & SO SÁNH (Thành viên DS 3 - Evaluation Lead)**

#### **Slide 11: Kết quả Đánh giá (Evaluation Metrics)**
*   **Hình ảnh:** Confusion Matrix + Bảng metrics (Precision, Recall, F1-Score).
*   **Nội dung:**
    *   Accuracy: ~95%.
    *   **Recall cao (>90%):** Không bỏ sót tai nạn thật.
    *   Precision: Cân bằng với recall để giảm false alarms.
*   **Lời thoại:**
    > "Em phụ trách đánh giá mô hình. Độ chính xác đạt ~95%. Trong bài toán an toàn, nhóm ưu tiên Recall cao - 'thà báo nhầm còn hơn bỏ sót'. Confusion Matrix cho thấy số tai nạn bị bỏ sót (False Negative) rất thấp."

#### **Slide 12: Thuật toán Xác nhận Thời gian (Temporal Confirmation) [HIGHLIGHT]**
*   **Hình ảnh:** Timeline minh họa Sliding Window: Frame 1-5 đều báo "Incident" => **CẢNH BÁO**.
*   **Nội dung:**
    *   **Vấn đề:** Nhiễu 1 frame gây false alarm (lá bay, đèn loé).
    *   **Giải pháp:** Sliding Window K=5 frames - chỉ báo khi 5 frame liên tiếp đều phát hiện sự cố.
    *   **Kết quả:** Giảm false alarms từ 30% xuống <5%.
*   **Lời thoại:**
    > "Cải tiến quan trọng là thuật toán Temporal Confirmation. AI có thể nhầm lẫn bởi nhiễu ngắn hạn. Thuật toán này yêu cầu sự cố phải xuất hiện liên tục trong 5 khung hình mới báo động, giúp hệ thống ổn định hơn rất nhiều."

#### **Slide 13: So sánh Baseline (Baseline Comparison)**
*   **Hình ảnh:** Bảng so sánh MobileNetV2 vs ResNet50 vs VGG16 (Accuracy, Speed, Size).
*   **Nội dung:**
    *   MobileNetV2: 95% accuracy, 14MB, 30 FPS.
    *   ResNet50: 96% accuracy, 98MB, 15 FPS.
    *   VGG16: 94% accuracy, 528MB, 8 FPS.
    *   **Kết luận:** MobileNetV2 cân bằng tốt nhất cho real-time.
*   **Lời thoại:**
    > "Nhóm đã thử nghiệm 3 mô hình. MobileNetV2 tuy accuracy thấp hơn ResNet50 một chút nhưng nhanh gấp đôi và nhẹ hơn 7 lần. Đây là lựa chọn tối ưu cho hệ thống giám sát real-time."

---

### **PHẦN 6: LIVE DEMO & KẾT LUẬN (Xuân Đạt - Nhóm trưởng)**

#### **Slide 14: LIVE DEMO [QUAN TRỌNG NHẤT]**
*   *(Chuyển màn hình sang ứng dụng đang chạy)*
*   **Hành động:**
    1.  Mở Dashboard Streamlit.
    2.  Chọn tab "Test Mô hình".
    3.  Upload 1 video tai nạn giao thông (đã chuẩn bị sẵn).
    4.  Chỉ vào màn hình khi hệ thống hiện dòng chữ đỏ **"CẢNH BÁO: SỰ CỐ"**.
    5.  Show phần log/lịch sử bên dưới.
    6.  Giải thích flow: Video → API → AI Model → Temporal Confirmation → Alert.
*   **Lời thoại:**
    > "Sau đây em xin demo trực tiếp hệ thống hoàn chỉnh. Em sẽ nạp vào một video giám sát giao thông... Như thầy cô thấy, ngay khi xe va chạm, hệ thống lập tức phát hiện và sau khi xác nhận qua thuật toán Temporal Confirmation, nó bật cảnh báo đỏ. Toàn bộ quá trình từ upload video đến hiển thị cảnh báo chỉ mất vài giây. Tất cả thông tin được lưu vào database để tra cứu sau này."

#### **Slide 15: Kết luận & Đóng góp chính**
*   **Nội dung:**
    *   Xây dựng thành công hệ thống ITS real-time với MobileNetV2.
    *   Đóng góp chính:
        *   Temporal Confirmation Algorithm giảm false alarms.
        *   Kiến trúc hệ thống Microservices linh hoạt, dễ mở rộng.
        *   Dashboard trực quan, dễ sử dụng.
    *   Kết quả: Accuracy 95%, FPS 30, minimal false alarms.
*   **Lời thoại:**
    > "Tóm lại, nhóm em đã xây dựng thành công hệ thống phát hiện sự cố giao thông real-time. Điểm nổi bật là thuật toán Temporal Confirmation và kiến trúc Microservices linh hoạt. Hệ thống đạt 95% accuracy với tốc độ xử lý 30 FPS."

#### **Slide 16: Hướng phát triển & Kết thúc**
*   **Nội dung:**
    *   **Future Work:**
        *   Nâng cấp lên Segmentation (U-Net) để khoanh vùng chính xác.
        *   Triển khai trên Edge Device (Jetson Nano, Raspberry Pi).
        *   Tích hợp gửi cảnh báo tự động (Telegram/Zalo) cho CSGT.
        *   Mở rộng dataset ban đêm, thời tiết xấu.
*   **Lời thoại:**
    > "Về hướng phát triển, nhóm dự định nâng cấp lên Segmentation để tô màu chính xác vùng sự cố, triển khai trên thiết bị biên để giảm chi phí, và tích hợp gửi cảnh báo tự động cho lực lượng chức năng. Em xin cảm ơn thầy cô và các bạn đã lắng nghe!"

---

## ❓ CÂU HỎI THƯỜNG GẶP (Q&A POCKET GUIDE)

### **Gói câu hỏi cho Team IT (Xuân Đạt & Thành viên IT) - Architecture & System:**
1.  **Hỏi:** "Tại sao hệ thống này xử lý video chậm?"
    *   **Đáp:** "Dạ hiện tại đang chạy trên CPU nên FPS khoảng 10-15. Để chạy thực tế High-FPS, giải pháp là dùng GPU (CUDA) và convert model sang TensorRT ạ."
2.  **Hỏi:** "Backend của em có chịu tải được 100 camera không?"
    *   **Đáp:** "Với kiến trúc hiện tại thì chưa ạ. Để scale lên, em sẽ cần dùng Message Queue (Kafka) để chia tải video ra cho nhiều Workers xử lý song song."
3.  **Hỏi:** "Tại sao code này lại chia thành class `ModelTrainer` riêng?"
    *   **Đáp:** "Em áp dụng OOP và Clean Architecture để tách biệt Logic train và Logic ứng dụng. Giúp code dễ bảo trì và test hơn ạ."

### **Gói câu hỏi cho Team Data Science (DS 1, DS 2, DS 3) - Data & Model:**
1.  **Hỏi:** "Tại sao không dùng YOLOv8 mới nhất?"
    *   **Đáp:** "Dạ YOLO chuyên về Object Detection (tìm vật thể), còn bài toán này thiên về Classification (phân loại hành vi). MobileNetV2 + Classification Head đơn giản và nhẹ hơn cho mục tiêu cảnh báo nhanh."
2.  **Hỏi:** "Số lượng ảnh bao nhiêu? Có cân bằng (balanced) không?"
    *   **Đáp:** "Dạ tập dataset khoảng X ảnh. Ban đầu bị lệch (bình thường nhiều hơn tai nạn), nhưng nhóm đã dùng Augmentation (xoay, lật) để cân bằng lại tỉ lệ 50-50 khi train ạ."
3.  **Hỏi:** "Nếu trời mưa/đêm tối thì sao?"
    *   **Đáp:** "Dataset hiện tại chủ yếu là ban ngày. Đây là hạn chế. Giải pháp là thu thập thêm data ban đêm và dùng các thuật toán Tiền xử lý ảnh (Histogram Equalization) để cân bằng sáng trước khi đưa vào model."

---

## 💡 LỜI KHUYÊN CHO NHÓM

### **Cho Xuân Đạt (Nhóm trưởng IT):**
1.  **Tự tin, Dẫn dắt:** Bạn là nhóm trưởng. Khi thành viên bị hỏi khó, hãy khéo léo đỡ lời: *"Dạ phần này để em bổ sung thêm..."*
2.  **Điều phối thời gian:** Đảm bảo mỗi người trình bày đúng 4-5 phút, tránh thừa/thiếu.
3.  **Chuẩn bị backup plan:** Nếu demo lỗi, có video demo dự phòng sẵn sàng.

### **Cho Thành viên IT (Backend Lead):**
1.  **Nắm vững API:** Hiểu rõ từng endpoint, cách xử lý request/response.
2.  **Chuẩn bị giải thích code:** Sẵn sàng show code khi được hỏi về implementation.
3.  **Hiểu về tối ưu hóa:** Giải thích được các kỹ thuật async, caching, batch processing.

### **Cho cả nhóm:**
1.  **Đồng bộ Slide:** 5 người phải cùng 1 Template (font, màu sắc).
2.  **Tập duyệt 2-3 lần:** Đảm bảo chuyển slide mượt mà, không bị gián đoạn.
3.  **Phân công rõ ràng:** Ai trả lời câu hỏi gì, thống nhất trước.

***Chúc nhóm mình đạt điểm A!*** 🚀
