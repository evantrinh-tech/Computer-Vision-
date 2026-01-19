# PHÂN CÔNG CÔNG VIỆC NHÓM & LÀM SLIDE THUYẾT TRÌNH
## ĐỀ TÀI: CHƯƠNG TRÌNH PHÁT HIỆN & PHÂN ĐOẠN HÀNH VI BẤT THƯỜNG (ITS)

### 1. CẤU TRÚC THÀNH VIÊN
*   **Tổng số:** 5 thành viên.
*   **Thành phần:** 2 Sinh viên CNTT (IT) + 3 Sinh viên Khoa học dữ liệu (DS).
*   **Nhóm trưởng:** Xuân Đạt (IT).

### 2. PHÂN CHIA VAI TRÒ CHUNG
*   **Nhóm CNTT (2 bạn):** Chịu trách nhiệm toàn bộ về **Hệ thống (System), Tích hợp (Integration), Giao diện (Frontend) & Triển khai (Deployment)**. Đảm bảo sản phẩm chạy Live mượt mà.
*   **Nhóm KHDL (3 bạn):** Chịu trách nhiệm trọn gói về **Dữ liệu & Mô hình AI (Data & Model)** cho 2 bài toán cốt lõi: Phát hiện (Detection) và Phân đoạn (Segmentation).

---

### 3. CHI TIẾT CÔNG VIỆC (TASK LIST)

#### 👤 THÀNH VIÊN 1 - XUÂN ĐẠT (CNTT 1 - Team Leader/System Architect)
*   **Vai trò:** Nhóm trưởng - Kiến trúc hệ thống & Backend/Core Logic.
*   **Nhiệm vụ chuyên môn:**
    *   Thiết kế kiến trúc Microservices/Modular.
    *   Xây dựng Pipeline xử lý video (Video Streaming Pipeline).
    *   Tối ưu hóa đa luồng (Multi-threading) để đảm bảo FPS cao.
    *   Tích hợp các Model AI vào hệ thống (Model Serving).
    *   Điều phối và giám sát tiến độ chung của nhóm.
*   **Nội dung Slide & Thuyết trình:**
    *   Sơ đồ khối kiến trúc hệ thống (System Architecture).
    *   Giải pháp kỹ thuật xử lý luồng (Tech Stack).
    *   Các kỹ thuật tối ưu hóa hiệu năng đã áp dụng.

#### 👤 THÀNH VIÊN 2 (CNTT 2 - Application Engineer & Demo Lead)
*   **Vai trò:** Phát triển ứng dụng (Frontend) & Triển khai (Deployment).
*   **Nhiệm vụ chuyên môn:**
    *   Xây dựng giao diện người dùng (Web App/Streamlit/Dashboard).
    *   Hiển thị trực quan kết quả (Bounding boxes, Segmentation masks) lên giao diện.
    *   Đóng gói ứng dụng (Docker/Executable) & Setup môi trường Demo.
    *   Quản lý kịch bản Demo trực tiếp (Live Demo).
*   **Nội dung Slide & Thuyết trình:**
    *   Giới thiệu tính năng ứng dụng (Features).
    *   Demo sản phẩm "sống" (Showcase).
    *   Hướng dẫn sử dụng & Triển khai.

#### 👤 THÀNH VIÊN 3 (DS 1 - Detection Lead)
*   **Vai trò:** Phụ trách bài toán Phát hiện vật thể/hành vi (Object Detection).
*   **Nhiệm vụ chuyên môn:**
    *   Thu thập, làm sạch & gán nhãn dữ liệu cho bài toán Detection.
    *   Huấn luyện & tinh chỉnh mô hình Detection (YOLO, MobileNet/SSD...).
    *   Đánh giá mô hình Detection (mAP, Precision, Recall).
*   **Nội dung Slide & Thuyết trình:**
    *   Tổng quan dữ liệu Detection (Dataset Overview).
    *   Kiến trúc & huấn luyện mô hình Detection.
    *   Kết quả đánh giá & Phân tích sai số (Detection Evaluation).

#### 👤 THÀNH VIÊN 4 (DS 2 - Segmentation Lead)
*   **Vai trò:** Phụ trách bài toán Phân đoạn (Segmentation).
*   **Nhiệm vụ chuyên môn:**
    *   Xử lý dữ liệu cho bài toán Segmentation (Pixel-level labeling/cleaning).
    *   Huấn luyện & tinh chỉnh mô hình Segmentation (U-Net, DeepLab...).
    *   Đánh giá mô hình Segmentation (IoU, Dice Coefficient).
*   **Nội dung Slide & Thuyết trình:**
    *   Chi tiết kỹ thuật Segmentation (U-Net...).
    *   Kết quả phân đoạn & Trực quan hóa (Masks visualization).

#### 👤 THÀNH VIÊN 5 (DS 3 - Model Comparison & Slide Master)
*   **Vai trò:** So sánh mô hình & Tổng hợp Slide.
*   **Nhiệm vụ chuyên môn:**
    *   So sánh hiệu năng giữa các mô hình Detection và Segmentation.
    *   Phân tích điểm mạnh, điểm yếu của từng mô hình.
    *   Đề xuất cải tiến và hướng phát triển.
*   **Nhiệm vụ Slide Master:**
    *   Gom nội dung từ 4 thành viên còn lại.
    *   Thiết kế Template, format font chữ, màu sắc đồng bộ.
    *   Viết phần: Giới thiệu chung, So sánh hiệu năng tổng thể, Kết luận & Hướng phát triển.
*   **Nội dung Slide & Thuyết trình:**
    *   Bảng so sánh hiệu năng các mô hình (Baseline Comparison).
    *   Phân tích kết quả tổng thể.
    *   Kết luận và hướng phát triển.

---

### 4. QUY TRÌNH PHỐI HỢP (WORKFLOW)

1.  **Giai đoạn 1 (Phân tách):**
    *   **IT Team:** Thống nhất API (Input/Output) giữa Backend (TV1) và Frontend (TV2).
    *   **DS Team:** Thống nhất format dữ liệu và cấu trúc Model để TV1 có thể tích hợp dễ dàng.
2.  **Giai đoạn 2 (Thực hiện & Slide nháp):**
    *   Mỗi thành viên hoàn thành task chuyên môn và chuẩn bị nội dung Slide thô (Text + Hình ảnh chèn vào PowerPoint/Google Slides nháp).
    *   TV 4 (Slide Master) tạo Template chung.
3.  **Giai đoạn 3 (Tích hợp & Hoàn thiện Slide):**
    *   TV 1 & TV 2 ghép code, chạy thử nghiệm hệ thống hoàn chỉnh.
    *   TV 4 thu thập slide của mọi người, ghép vào Template, căn chỉnh thẩm mỹ.
4.  **Giai đoạn 4 (Rehearsal):**
    *   TV 2 chuẩn bị máy Demo.
    *   Cả nhóm tập dượt theo kịch bản Slide đã chốt.

### 5. YÊU CẦU ĐẦU RA
*   **IT Team:** Phải có **Sản phẩm chạy thật** (Live Demo) mượt mà, giao diện thân thiện, không lỗi vặt khi thuyết trình.
*   **DS Team:** Phải có **Số liệu đánh giá** (Metrics) rõ ràng và **Hình ảnh trực quan** (Charts, Visualized Images) chứng minh hiệu quả mô hình.
