# 🎬 KỊCH BẢN THUYẾT TRÌNH & NỘI DUNG SLIDE CHI TIẾT
**Dự án:** ITS - Phát hiện sự cố giao thông (Detecting & Segmenting Abnormal Behavior)
**Thời lượng:** 20-25 phút
**Nhóm:** 6 thành viên (5 Data Science (DS), 1 IT Leader)

---

## 📅 BẢNG PHÂN CÔNG TỔNG QUÁT

| STT | Người trình bày | Vai trò | Nội dung chính |
|:---:|:--- |:--- |:--- |
| **1** | **Thành viên DS 1** | Mở đầu | Lý do chọn đề tài, Mục tiêu, Tổng quan Dữ liệu. |
| **2** | **Thành viên DS 2** | Xử lý dữ liệu | Tiền xử lý (Resize/Norm), Data Augmentation. |
| **3** | **Thành viên DS 3** | Modeling | CNN, Transfer Learning, MobileNetV2 Architecture. |
| **4** | **Thành viên DS 4** | Training | Quá trình huấn luyện, Hyperparameters, Loss/Acc charts. |
| **5** | **Thành viên DS 5** | Evaluation | Metrics (F1/Recall), Confusion Matrix, **Temporal Algorithm**. |
| **6** | **Trưởng nhóm (IT)** | System & Demo | Kiến trúc hệ thống, Tech stack, **Live Demo**, Q&A. |

---

## 📝 CHI TIẾT TỪNG PHẦN (Slide & Lời thoại)

### **PHẦN 1: MỞ ĐẦU & DỮ LIỆU (Thành viên DS 1)**

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
    > "Trong thời đại smart city, camera có ở khắp nơi. Tuy nhiên, việc giám sát thủ công 24/7 là bất khả thi. Mục tiêu của nhóm là xây dựng một 'đôi mắt ảo' giúp tự động phát hiện tai nạn hoặc sự cố ngay khi nó xảy ra."

#### **Slide 3: Tổng quan Dữ liệu (Dataset)**
*   **Hình ảnh:** Biểu đồ tròn phân bố (Normal vs Incident). Một vài ảnh mẫu (Sample images) của từng loại.
*   **Nội dung:**
    *   Nguồn: Thu thập từ Youtube, Dataset công khai (AI City Challenge...).
    *   Class 1: **Normal** (Giao thông bình thường).
    *   Class 2: **Incident** (Tai nạn, cháy, va chạm).
    *   Khó khăn: Ảnh mờ, góc quay đa dạng.
*   **Lời thoại:**
    > "Để dạy cho máy biết thế nào là sự cố, chúng em đã thu thập và gán nhãn dữ liệu thành 2 loại: Bình thường và Sự cố. Dữ liệu bao gồm nhiều bối cảnh khác nhau từ cao tốc đến ngã tư."

---

### **PHẦN 2: XỬ LÝ DỮ LIỆU (Thành viên DS 2)**

#### **Slide 4: Tiền xử lý (Preprocessing)**
*   **Hình ảnh:** Sơ đồ: Ảnh gốc -> Resize (224x224) -> Normalize (0-1).
*   **Nội dung:**
    *   Resize: Đưa về chuẩn đầu vào của MobileNetV2 (224x224).
    *   Normalization: Chia giá trị pixel cho 255 để đưa về khoảng [0, 1].
*   **Lời thoại:**
    > "Ảnh từ camera có kích thước lộn xôn. Bước đầu tiên là chuẩn hóa chúng về kích thước 224x224 để phù hợp với mô hình, đồng thời chuẩn hóa giá trị pixel để mô hình hội tụ nhanh hơn."

#### **Slide 5: Data Augmentation (Tăng cường dữ liệu)**
*   **Hình ảnh:** 1 ảnh gốc ở giữa -> Mũi tên ra 4 ảnh biến thể (Xoay nghiêng, lật ngang, tối hơn, zoom).
*   **Nội dung:**
    *   Kỹ thuật: Rotation, Horizontal Flip, Brightness adjust.
    *   Mục đích: Giảm Overfitting, giúp model học tốt trong điều kiện nắng/mưa.
*   **Lời thoại:**
    > "Vì dữ liệu sự cố rất hiếm, nhóm dùng kỹ thuật Data Augmentation. Từ một ảnh tai nạn, chúng em tạo ra nhiều phiên bản: xoay, lật, chỉnh độ sáng... Điều này giúp AI không 'học vẹt' mà hiểu bản chất vấn đề, nhận diện tốt cả khi trời tối hay camera bị rung."

---

### **PHẦN 3: MÔ HÌNH HÓA (Thành viên DS 3)**

#### **Slide 6: Tại sao chọn CNN & Transfer Learning?**
*   **Hình ảnh:** Sơ đồ ý tưởng Transfer Learning (ImageNet Knowledge -> ITS Task).
*   **Nội dung:**
    *   CNN: Chuyên trị xử lý ảnh (Feature Extraction).
    *   Transfer Learning: Tận dụng model đã train sẵn (Pre-trained) để tiết kiệm thời gian và tăng độ chính xác.
*   **Lời thoại:**
    > "Thay vì xây dựng mô hình từ con số 0 cần hàng triệu ảnh, nhóm áp dụng Transfer Learning. Chúng em sử dụng 'trí tuệ' của các mô hình lớn đã học hàng triệu vật thể, sau đó tinh chỉnh lại để chuyên phát hiện sự cố giao thông."

#### **Slide 7: Kiến trúc MobileNetV2 (Architecture)**
*   **Hình ảnh:** Sơ đồ kiến trúc [Input -> MobileNetV2 (Frozen) -> GlobalAvgPool -> Dense -> Dropout -> Output].
*   **Nội dung:**
    *   **Base Model:** MobileNetV2 (nhẹ, nhanh, phù hợp realtime). So sánh: MobileNetV2 (14MB) vs VGG16 (500MB+).
    *   **Custom Head:** Thêm các lớp Dense để phân loại 2 class.
*   **Lời thoại:**
    > "Nhóm quyết định chọn MobileNetV2 làm xương sống (Backbone). Lý do là nó cực kỳ nhẹ và nhanh, rất phù hợp cho bài toán giám sát thời gian thực (Real-time). Chúng em giữ lại phần trích xuất đặc trưng và chỉ thay thế phần đuôi để phân loại: Có sự cố hay Không."

---

### **PHẦN 4: HUẤN LUYỆN (Thành viên DS 4)**

#### **Slide 8: Cấu hình Huấn luyện (Training Config)**
*   **Nội dung (Bảng):**
    *   Framework: TensorFlow/Keras.
    *   Optimizer: Adam (Learning rate = 0.001).
    *   Loss Function: Binary Crossentropy.
    *   Epochs: 20-50.
    *   Hardware: Google Colab GPU / Local GPU.
*   **Lời thoại:**
    > "Chúng em huấn luyện mô hình sử dụng TensorFlow. Hàm tối ưu Adam được chọn vì sự ổn định. Quá trình train được thực hiện trên GPU để tăng tốc độ xử lý."

#### **Slide 9: Biểu đồ Loss & Accuracy**
*   **Hình ảnh:** 2 biểu đồ đường (Line chart).
    *   Trục X: Epochs.
    *   Trục Y: Loss/Accuracy.
    *   Đường Train và Validation đi sát nhau (Good fit).
*   **Lời thoại:**
    > "Đây là kết quả huấn luyện. Đường màu xanh là Train, màu cam là Validation. Thầy cô có thể thấy Loss giảm đều và Accuracy tăng dần lên mức 9x%, chứng tỏ mô hình học tốt và không bị Overfitting quá mức."

---

### **PHẦN 5: ĐÁNH GIÁ & THUẬT TOÁN (Thành viên DS 5)**

#### **Slide 10: Kết quả Đánh giá (Metrics)**
*   **Hình ảnh:** Confusion Matrix (Ma trận nhầm lẫn).
*   **Nội dung:**
    *   Accuracy: ~95% (Ví dụ).
    *   **Precision/Recall:** Nhấn mạnh tầm quan trọng của Recall (Không được bỏ sót tai nạn).
*   **Lời thoại:**
    > "Độ chính xác tổng thể đạt X%. Tuy nhiên, trong bài toán an toàn này, nhóm ưu tiên chỉ số Recall - tức là 'thà báo nhầm còn hơn bỏ sót'. Nhìn vào Confusion Matrix, số lượng tai nạn bị bỏ sót (False Negative) là rất thấp."

#### **Slide 11: Thuật toán Xác nhận theo thời gian (Temporal Confirmation) [HIGHLIGHT]**
*   **Hình ảnh:** Minh họa Timeline. Frame 1 (Báo) -> Frame 2 (Báo) ... -> Frame K (Báo) => **ALARM ON**.
*   **Nội dung:**
    *   Vấn đề: Nhiễu, nháy (Flickering) trong 1 frame đơn lẻ.
    *   Giải pháp: Sliding Window K=5 frames.
    *   Kết quả: Giảm báo động giả (False Alarms).
*   **Lời thoại:**
    > "Một cải tiến quan trọng của nhóm là thuật toán 'Xác nhận theo thời gian'. AI thường bị 'giật mình' bởi lá cây bay hay ánh đèn loé. Thuật toán này yêu cầu sự cố phải tồn tại liên tục trong ít nhất 5 khung hình thì hệ thống mới phát cảnh báo. Điều này giúp hệ thống hoạt động ổn định hơn rất nhiều."

---

### **PHẦN 6: HỆ THỐNG & DEMO (Trưởng nhóm IT)**

#### **Slide 12: Kiến trúc Hệ thống (System Overview)**
*   **Hình ảnh:** Sơ đồ khối:
    *   [Camera/Video File] ---> [API Server (FastAPI)] ---> [AI Engine (MobileNetV2 + Temporal Algo)] ---> [Database (PostgreSQL)] ---> [Dashboard (Streamlit)].
*   **Lời thoại:**
    > "Để đưa mô hình vào thực tế, em đã xây dựng một kiến trúc 3 lớp. Backend sử dụng FastAPI xử lý bất đồng bộ để đảm bảo tốc độ. Mô hình AI được nhúng trực tiếp vào luồng xử lý video. Kết quả nhận diện được lưu vào Database và hiển thị tức thì lên Dashboard."

#### **Slide 13: Công nghệ sử dụng (Tech Stack)**
*   **Hình ảnh:** Logo các công nghệ: Python, TensorFlow, FastAPI, Streamlit, MLflow, OpenCV.
*   **Lời thoại:**
    > "Đây là bộ công nghệ (Stack) nhóm sử dụng. FastAPI cho hiệu năng cao, Streamlit giúp người vận hành dễ dàng theo dõi, và MLflow để quản lý các phiên bản mô hình."

#### **Slide 14: LIVE DEMO [QUAN TRỌNG NHẤT]**
*   *(Chuyển màn hình sang ứng dụng đang chạy)*
*   **Hành động:**
    1.  Mở Dashboard Streamlit.
    2.  Chọn tab "Test Mô hình".
    3.  Upload 1 video tai nạn giao thông (đã chuẩn bị sẵn).
    4.  Chỉ vào màn hình khi hệ thống hiện dòng chữ đỏ **"CẢNH BÁO: SỰ CỐ"**.
    5.  Show phần log/lịch sử bên dưới.
*   **Lời thoại:**
    > "Sau đây em xin demo trực tiếp. Em sẽ nạp vào hệ thống một video giám sát... Như thầy cô thấy, ngay khi xe va chạm, hệ thống lập tức khoanh vùng và bật cảnh báo đỏ. Độ trễ xử lý chỉ khoảng vài mili-giây."

#### **Slide 15: Hướng phát triển & Kết luận**
*   **Nội dung:**
    *   Sử dụng YOLO/Mask R-CNN để khoanh vùng (segmentation) chính xác hơn (Future work).
    *   Triển khai Edge Device (Jetson Nano).
    *   Tích hợp gửi tin nhắn Telegram/Zalo cho CSGT.
*   **Lời thoại:**
    > "Trong tương lai, nhóm dự định nâng cấp lên Segmentation để tô màu chính xác vùng tai nạn và tích hợp gửi tin nhắn cảnh báo tự động cho lực lượng chức năng. Em xin cảm ơn thầy cô đã lắng nghe!"

---

## ❓ CÂU HỎI THƯỜNG GẶP (Q&A POCKET GUIDE)

### **Gói câu hỏi cho Trưởng nhóm IT (Architecture & Code):**
1.  **Hỏi:** "Tại sao hệ thống này xử lý video chậm?"
    *   **Đáp:** "Dạ hiện tại đang chạy trên CPU nên FPS khoảng 10-15. Để chạy thực tế High-FPS, giải pháp là dùng GPU (CUDA) và convert model sang TensorRT ạ."
2.  **Hỏi:** "Backend của em có chịu tải được 100 camera không?"
    *   **Đáp:** "Với kiến trúc hiện tại thì chưa ạ. Để scale lên, em sẽ cần dùng Message Queue (Kafka) để chia tải video ra cho nhiều Workers xử lý song song."
3.  **Hỏi:** "Tại sao code này lại chia thành class `ModelTrainer` riêng?"
    *   **Đáp:** "Em áp dụng OOP và Clean Architecture để tách biệt Logic train và Logic ứng dụng. Giúp code dễ bảo trì và test hơn ạ."

### **Gói câu hỏi cho Team Data Science (Model & Math):**
1.  **Hỏi:** "Tại sao không dùng YOLOv8 mới nhất?"
    *   **Đáp:** "Dạ YOLO chuyên về Object Detection (tìm vật thể), còn bài toán này thiên về Classification (phân loại hành vi). MobileNetV2 + Classification Head đơn giản và nhẹ hơn cho mục tiêu cảnh báo nhanh."
2.  **Hỏi:** "Số lượng ảnh bao nhiêu? Có cân bằng (balanced) không?"
    *   **Đáp:** "Dạ tập dataset khoảng X ảnh. Ban đầu bị lệch (bình thường nhiều hơn tai nạn), nhưng nhóm đã dùng Augmentation (xoay, lật) để cân bằng lại tỉ lệ 50-50 khi train ạ."
3.  **Hỏi:** "Nếu trời mưa/đêm tối thì sao?"
    *   **Đáp:** "Dataset hiện tại chủ yếu là ban ngày. Đây là hạn chế. Giải pháp là thu thập thêm data ban đêm và dùng các thuật toán Tiền xử lý ảnh (Histogram Equalization) để cân bằng sáng trước khi đưa vào model."

---

## 💡 LỜI KHUYÊN CHO NHÓM TRƯỞNG
1.  **Tự tin, Dẫn dắt:** Bạn là người "cầm trịch". Khi thành viên team DS bị thầy hỏi khó (bí), hãy khéo léo đỡ lời: *"Dạ phần kỹ thuật này để em bổ sung thêm cho bạn..."*
2.  **Chuẩn bị Demo kỹ:** File video demo phải test trước 10 lần. Đảm bảo nó chạy mượt, không lỗi. Nên quay sẵn 1 video dự phòng (backup) lỡ lúc demo máy bị treo.
3.  **Đồng bộ Slide:** Slide của 6 người phải cùng 1 Template (font chữ, màu sắc). Đừng để mỗi người 1 kiểu.

***Chúc nhóm mình đạt điểm A!*** 🚀
