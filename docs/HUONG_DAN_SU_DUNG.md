# 📖 HƯỚNG DẪN SỬ DỤNG HỆ THỐNG ITS

Tài liệu này hướng dẫn chi tiết cách cài đặt, vận hành và sử dụng các chức năng của hệ thống ITS.

---

## 1. Yêu Cầu Hệ Thống
*   **Hệ điều hành:** Windows 10/11 (Khuyến nghị), Linux, MacOS.
*   **Python:** Phiên bản 3.10 hoặc 3.11.
*   **RAM:** Tối thiểu 8GB (Khuyến nghị 16GB nếu train model).
*   **GPU:** Có NVIDIA GPU là lợi thế (train nhanh hơn), nếu không vẫn chạy được trên CPU.

## 2. Cài Đặt (Setup)

### Bước 1: Clone Model & Dữ Liệu
Đảm bảo bạn đã có đầy đủ source code và dữ liệu.
*   Folder `data/images/normal`: Chứa ảnh giao thông bình thường.
*   Folder `data/images/incident`: Chứa ảnh có sự cố.

### Bước 2: Khởi Tạo Môi Trường
Chạy file script tự động (Windows):
```bat
scripts\tao_venv.bat
```
Script này sẽ:
1.  Tạo môi trường ảo `venv311`.
2.  Tự động cài đặt các thư viện từ `requirements.txt`.

Nếu cài thủ công:
```bash
python -m venv venv311
.\venv311\Scripts\activate
pip install -r requirements.txt
```

---

## 3. Vận Hành Hệ Thống

Bạn có thể sử dụng Menu tự động hoặc chạy lệnh thủ công.

### Cách 1: Dùng Menu (Dễ nhất)
Chạy file `he_thong.bat` (Click đúp chuột).
Một menu sẽ hiện ra với các lựa chọn:
*   `[1] Mở Dashboard`: Chạy giao diện Web Streamlit.
*   `[2] Chạy API`: Khởi động Backend Server.
*   `[3] Huấn luyện Model`: Tự động chạy script training.
*   `[4] Test Camera`: Chạy demo nhận diện từ Webcam/Video.

### Cách 2: Chạy Lệnh Thủ Công

#### 🖥️ A. Chạy Giao Diện Web (Dashboard)
```bash
.\venv311\Scripts\streamlit run app.py
```
Giao diện sẽ mở tại: `http://localhost:8501`

#### 🧠 B. Huấn Luyện Mô Hình
```bash
.\venv311\Scripts\python train_cnn.py
```
*   Quá trình train sẽ tự động load ảnh từ `data/`, chia tập train/test/val.
*   Kết quả model lưu tại: `models/CNN_model/model.keras`.
*   Biểu đồ kết quả log tại: `logs/`.

#### 🔌 C. Chạy API Server (Backend)
```bash
.\venv311\Scripts\python start_api.py
```
*   API chạy tại: `http://localhost:8000`.
*   Tài liệu API (Swagger): `http://localhost:8000/docs`.

---

## 4. Các Chức Năng Trên Dashboard

### 🏠 Tab Trang Chủ
*   Xem thống kê tổng quan về hệ thống: số lượng data, cấu hình hiện tại.

### 📊 Tab Xem Dữ Liệu
*   Duyệt xem các ảnh trong tập dữ liệu.
*   Xem phân bố số lượng Normal vs Incident.

### 🎓 Tab Huấn Luyện
*   **Epochs:** Chọn số vòng lặp huấn luyện (Default: 10-20).
*   **Batch Size:** Số ảnh học một lần (Default: 32).
*   Bấm **"Bắt đầu huấn luyện"** để train lại model trực tiếp trên web.

### 🔍 Tab Test Mô Hình
*   **Upload Ảnh:** Chọn file `.jpg`, `.png` test thử.
*   **Upload Video:** Chọn file `.mp4`. Hệ thống sẽ scan từng frame và cảnh báo nếu có sự cố.

---

## 5. Xử Lý Sự Cố Thường Gặp

**Lỗi: `ModuleNotFoundError: No module named 'tensorflow'`**
*   **Nguyên nhân:** Chưa kích hoạt môi trường ảo.
*   **Khắc phục:** Hãy chắc chắn bạn chạy lệnh thông qua `venv311\Scripts\python` hoặc đã chạy `venv311\Scripts\activate` trước.

**Lỗi: `cudaGetDevice() failed` (TensorFlow)**
*   **Nguyên nhân:** Máy không có GPU NVIDIA hoặc chưa cài CUDA.
*   **Khắc phục:** Hệ thống sẽ tự động chuyển về chạy CPU. Đây chỉ là warning, không phải lỗi nghiêm trọng.

**Lỗi: Không load được ảnh**
*   **Khắc phục:** Kiểm tra lại đường dẫn `data/images`. Tên folder phải chính xác là `normal` và `incident`.

---
*Chúc bạn sử dụng hệ thống hiệu quả!*
