# 🏗️ KIẾN TRÚC HỆ THỐNG ITS

## Sơ Đồ Tổng Quan (High-Level Architecture)

Hệ thống được thiết kế theo mô hình **Software 1.0 + Software 2.0** hybrid.
*   **Software 1.0 (Code truyền thống):** Web Interface, API, Database logic.
*   **Software 2.0 (AI Model):** Neural Network học từ dữ liệu.

```mermaid
graph TD
    User[Người dùng] -->|Tương tác| Streamlit[Dashboard (Frontend)]
    Streamlit -->|Gọi hàm| AI_Engine[AI Processing Core]
    Streamlit -->|Xem dữ liệu| DB[(Database)]
    
    subgraph "AI Core Layer"
    AI_Engine -->|1. Tiền xử lý| Preprocessing[Resize/Normalize]
    Preprocessing -->|2. Dự đoán| CNN[MobileNetV2 Model]
    CNN -->|3. Hậu xử lý| TempleAlgo[Temporal Confirmation]
    end
    
    Camera[Camera/Video Nguồn] -->|Stream| Streamlit
```

---

## 1. Data Layer (Tầng Dữ Liệu)
*   **Raw Data:** Ảnh/Video từ camera giám sát.
*   **Processed Data:** Ảnh đã resize (224x224), chuẩn hóa pixel.
*   **Database:** PostgreSQL (hoặc SQLite cho demo).
    *   Lưu trữ metadata về các sự cố phát hiện được (thời gian, địa điểm, độ tin cậy).

## 2. Model Layer (Tầng Mô Hình - AI)
Cốt lõi trí tuệ của hệ thống.
*   **Backbone:** MobileNetV2 (Pre-trained on ImageNet).
    *   Nhiệm vụ: Trích xuất đặc trưng (cạnh, góc, hình khối) từ ảnh.
*   **Head:** Custom Dense Library.
    *   Nhiệm vụ: Phân loại đặc trưng thành 2 lớp (Normal/Incident).
*   **Training Strategy:** Transfer Learning + Fine-tuning.

## 3. Application Layer (Tầng Ứng Dụng)
*   **Backend (FastAPI):**
    *   Xử lý các request suy luận (inference).
    *   Quản lý hàng đợi (queue) nếu có nhiều request cùng lúc.
*   **Frontend (Streamlit):**
    *   Visualize kết quả realtime.
    *   Hiển thị biểu đồ thống kê.

## 4. Infrastructure & Deployment (Tầng Hạ Tầng)
*   **Environment:** Python Virtual Environment (`venv311`).
*   **Dependency Management:** `requirements.txt`.
*   **OS:** Cross-platform (Windows, Linux).

---

## Luồng Xử Lý Dữ Liệu (Data Flow)

1.  **Input:** Video stream từ camera.
2.  **Frame Extraction:** Tách video thành các frame ảnh rời rạc (ví dụ 10 FPS).
3.  **Preprocessing:** Resize ảnh về 224x224.
4.  **Inference:**
    *   Model CNN tính toán xác suất (Probability) sự cố cho từng frame.
    *   Ví dụ: Frame 1 (0.1), Frame 2 (0.8), Frame 3 (0.9)...
5.  **Temporal Confirmation:**
    *   Thuật toán gom nhóm các frame liên tiếp.
    *   Chỉ khi xác suất cao xuất hiện liên tục -> Kích hoạt **Event**.
6.  **Alert:**
    *   Lưu event vào Database.
    *   Hiển thị cảnh báo đỏ trên Dashboard.

---
*Tài liệu này cung cấp cái nhìn toàn cảnh về kỹ thuật cho Developer và Architect.*
