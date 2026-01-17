import streamlit as st
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from PIL import Image
import io
import time

if sys.platform == 'win32' and 'streamlit' not in sys.modules:
    try:
        import io
        if not isinstance(sys.stdout, io.TextIOWrapper):
            sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
        if not isinstance(sys.stderr, io.TextIOWrapper):
            sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
    except:
        pass

# Thêm thư mục gốc vào path để import modules
sys.path.insert(0, str(Path(__file__).parent))

# Cấu hình trang Streamlit (Tiêu đề, icon, layout rộng)
st.set_page_config(
    page_title="Hệ thống Phát hiện Sự cố Giao thông",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

if 'training_in_progress' not in st.session_state:
    st.session_state.training_in_progress = False
if 'training_results' not in st.session_state:
    st.session_state.training_results = None
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False

def load_image_files(folder_path: Path):
    """
    Hàm tiện ích để load danh sách file ảnh trong folder.
    Hỗ trợ các đuôi file: .jpg, .jpeg, .png, .webp, .gif
    """
    extensions = ['*.jpg', '*.jpeg', '*.png', '*.webp', '*.gif']
    image_files = []
    for ext in extensions:
        image_files.extend(list(folder_path.glob(ext)))
        image_files.extend(list(folder_path.glob(ext.upper())))
    return sorted(image_files)

def get_image_count():

    data_path = Path("data/images")
    normal_dir = data_path / "normal"
    incident_dir = data_path / "incident"

    normal_count = 0
    incident_count = 0

    if normal_dir.exists():
        normal_count = len(load_image_files(normal_dir))
    if incident_dir.exists():
        incident_count = len(load_image_files(incident_dir))

    return normal_count, incident_count

try:
    st.sidebar.title(" Hệ thống Phát hiện Sự cố")
    st.sidebar.markdown("---")

    # Menu điều hướng dùng Radio Button
    page = st.sidebar.radio(
        "Chọn chức năng:",
        [" Trang chủ", " Xem dữ liệu", " Huấn luyện mô hình", " Test mô hình", " Kết quả & Metrics"]
    )
except Exception as e:
    # Fallback nếu lỗi sidebar
    st.error(f"Lỗi khi khởi tạo sidebar: {e}")
    st.exception(e)
    page = " Trang chủ"

if page == " Trang chủ":
    st.markdown('<div class="main-header"> Hệ thống Phát hiện Sự cố Giao thông</div>', unsafe_allow_html=True)
    st.markdown("---")

    col1, col2, col3 = st.columns(3)

    normal_count, incident_count = get_image_count()
    total_images = normal_count + incident_count

    with col1:
        st.metric("📁 Tổng số ảnh", total_images)
    with col2:
        st.metric(" Ảnh bình thường", normal_count)
    with col3:
        st.metric("⚠️ Ảnh có sự cố", incident_count)

    st.markdown("---")

    st.markdown("### 📋 Tổng quan hệ thống")

    st.markdown("###  Hướng dẫn sử dụng")

    model_path = Path("models/CNN_model")
    if model_path.exists():
        st.success(" Đã có mô hình được huấn luyện")
    else:
        st.warning("⚠️ Chưa có mô hình. Vui lòng huấn luyện mô hình trước khi sử dụng.")

elif page == " Xem dữ liệu":
    st.title(" Xem dữ liệu")
    st.markdown("---")

    data_path = Path("data/images")
    normal_dir = data_path / "normal"
    incident_dir = data_path / "incident"

    tab1, tab2 = st.tabs([" Ảnh bình thường", "⚠️ Ảnh có sự cố"])

    with tab1:
        st.subheader("Ảnh bình thường (Normal)")

        if not normal_dir.exists():
            st.error(f"❌ Không tìm thấy folder: {normal_dir}")
        else:
            image_files = load_image_files(normal_dir)

            if not image_files:
                st.warning("⚠️ Không tìm thấy ảnh trong folder này")
            else:
                st.info(f"📁 Tìm thấy {len(image_files)} ảnh")

                cols_per_row = 3
                for i in range(0, len(image_files), cols_per_row):
                    cols = st.columns(cols_per_row)
                    for j, col in enumerate(cols):
                        if i + j < len(image_files):
                            img_file = image_files[i + j]
                            try:
                                img = Image.open(img_file)
                                col.image(img, caption=img_file.name, use_container_width=True)
                            except Exception as e:
                                col.error(f"Không thể load: {img_file.name}")

    with tab2:
        st.subheader("Ảnh có sự cố (Incident)")

        if not incident_dir.exists():
            st.error(f"❌ Không tìm thấy folder: {incident_dir}")
        else:
            image_files = load_image_files(incident_dir)

            if not image_files:
                st.warning("⚠️ Không tìm thấy ảnh trong folder này")
            else:
                st.info(f"📁 Tìm thấy {len(image_files)} ảnh")

                cols_per_row = 3
                for i in range(0, len(image_files), cols_per_row):
                    cols = st.columns(cols_per_row)
                    for j, col in enumerate(cols):
                        if i + j < len(image_files):
                            img_file = image_files[i + j]
                            try:
                                img = Image.open(img_file)
                                col.image(img, caption=img_file.name, use_container_width=True)
                            except Exception as e:
                                col.error(f"Không thể load: {img_file.name}")

elif page == " Huấn luyện mô hình":
    st.title(" Huấn luyện mô hình CNN")
    st.markdown("---")

    normal_count, incident_count = get_image_count()
    total_images = normal_count + incident_count

    if total_images == 0:
        st.error("❌ Không tìm thấy ảnh để huấn luyện!")
        st.info("Vui lòng đảm bảo có ảnh trong `data/images/normal/` và `data/images/incident/`")
    else:
        st.success(f" Tìm thấy {normal_count} ảnh bình thường và {incident_count} ảnh có sự cố")

        st.markdown("###  Cấu hình huấn luyện")

        col1, col2 = st.columns(2)

        with col1:
            epochs = st.number_input("Số epochs", min_value=1, max_value=200, value=50, step=5)
            batch_size = st.number_input("Batch size", min_value=1, max_value=128, value=32, step=8)

        with col2:
            use_transfer_learning = st.checkbox("Sử dụng Transfer Learning", value=True)
            image_size = st.selectbox("Kích thước ảnh", [(224, 224), (256, 256), (128, 128)], format_func=lambda x: f"{x[0]}x{x[1]}")

        if st.button(" Bắt đầu huấn luyện", type="primary", use_container_width=True):
            if st.session_state.training_in_progress:
                st.warning("⚠️ Đang có quá trình huấn luyện khác đang chạy!")
            else:
                st.session_state.training_in_progress = True

                # Container để hiển thị log quá trình training
                status_container = st.container()
                progress_bar = st.progress(0)

                with status_container:
                    try:
                        status_text = st.empty()
                        status_text.info("📦 Đang import các thư viện (TensorFlow, MLflow...) - Có thể mất 10-30 giây")
                        progress_bar.progress(10)

                        from src.training.trainer import ModelTrainer
                        from src.utils.config import settings
                        import mlflow

                        progress_bar.progress(20)
                        status_text.info(" Đã import xong các thư viện")

                        progress_bar.progress(30)
                        status_text.info(" Đang load cấu hình...")
                        config_path = Path("configs/training_config.yaml")
                        if not config_path.exists():
                            config_path = None
                            status_text.info("ℹ️ Sử dụng cấu hình mặc định")

                        progress_bar.progress(40)

                        status_text.info(" Đang khởi tạo trainer...")
                        try:
                            trainer = ModelTrainer(model_type='CNN', config_path=config_path)
                            if trainer.use_mlflow:
                                status_text.info(" Trainer đã sẵn sàng (MLflow tracking: ON)")
                            else:
                                status_text.info(" Trainer đã sẵn sàng (MLflow tracking: OFF - training vẫn hoạt động bình thường)")
                        except Exception as trainer_error:
                            status_text.error(f"❌ Lỗi khi khởi tạo trainer: {trainer_error}")
                            raise

                        progress_bar.progress(60)
                        status_text.success(" Đã khởi tạo trainer thành công!")

                        if use_transfer_learning:
                            trainer.config['model'] = trainer.config.get('model', {})
                            trainer.config['model']['use_transfer_learning'] = True
                            trainer.config['model']['image_size'] = list(image_size)

                        trainer.config['training'] = trainer.config.get('training', {})
                        trainer.config['training']['epochs'] = epochs
                        trainer.config['training']['batch_size'] = batch_size

                        status_text.info(" Đang chuẩn bị dữ liệu (load và xử lý ảnh)...")
                        progress_bar.progress(65)

                        data_path = Path("data/images")
                        # Chia dataset: 70% train, 10% val, 20% test
                        X_train, y_train, X_val, y_val, X_test, y_test = trainer.prepare_data(
                            data_path=data_path,
                            test_size=0.2,
                            val_size=0.1
                        )

                        progress_bar.progress(70)
                        status_text.success(f" Đã chuẩn bị: {len(X_train)} train, {len(X_val)} val, {len(X_test)} test")

                        status_text.info(" Đang huấn luyện mô hình... (Quá trình này có thể mất 10-30 phút tùy vào số lượng ảnh)")
                        progress_bar.progress(75)

                        training_results = trainer.train(
                            X_train, y_train,
                            X_val, y_val,
                            run_name=f"CNN_training_{int(time.time())}"
                        )

                        progress_bar.progress(90)

                        st.info(" Đang đánh giá mô hình...")
                        test_metrics = trainer.evaluate_on_test(X_test, y_test)

                        progress_bar.progress(100)
                        status_text.text(" Hoàn tất!")

                        st.session_state.training_results = {
                            'train_metrics': training_results.get('train_metrics', {}),
                            'val_metrics': training_results.get('val_metrics', {}),
                            'test_metrics': test_metrics,
                            'model_path': str(training_results.get('model_path', ''))
                        }
                        st.session_state.model_loaded = True
                        st.session_state.training_in_progress = False

                        st.success("🎉 Huấn luyện hoàn tất!")

                        st.markdown("###  Kết quả huấn luyện")

                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Train Accuracy", f"{training_results.get('train_metrics', {}).get('accuracy', 0):.4f}")
                        with col2:
                            st.metric("Val Accuracy", f"{training_results.get('val_metrics', {}).get('accuracy', 0):.4f}")
                        with col3:
                            st.metric("Test Accuracy", f"{test_metrics.get('accuracy', 0):.4f}")

                    except Exception as e:
                        st.error(f"❌ Lỗi: {str(e)}")
                        st.exception(e)
                        st.session_state.training_in_progress = False

elif page == " Test mô hình":
    st.title(" Test mô hình")
    st.markdown("---")

    model_path = Path("models/CNN_model")

    if not model_path.exists():
        st.warning("⚠️ Chưa có mô hình được huấn luyện!")
        st.info("Vui lòng huấn luyện mô hình trước (trang 'Huấn luyện mô hình')")
    else:
        st.success(" Đã tìm thấy mô hình (Ready to predict)")

        try:
            from src.models.cnn import CNNModel
            from src.data_processing.image_processor import ImageProcessor

            if 'cnn_model' not in st.session_state:
                with st.spinner("Đang load mô hình..."):
                    model = CNNModel()
                    model.load(model_path)
                    st.session_state.cnn_model = model
                    st.session_state.image_processor = ImageProcessor(image_size=(224, 224))

            st.markdown("### 📤 Upload ảnh để test")

            uploaded_file = st.file_uploader(
                "Chọn ảnh",
                type=['jpg', 'jpeg', 'png', 'webp'],
                help="Upload ảnh từ camera giao thông để kiểm tra"
            )

            if uploaded_file is not None:
                img = Image.open(uploaded_file)
                st.image(img, caption="Ảnh đã upload", use_container_width=True)

                if st.button(" Phân tích ảnh", type="primary"):
                    with st.spinner("Đang xử lý..."):
                        try:
                            temp_path = Path("temp_upload.jpg")
                            img.save(temp_path)

                            image = st.session_state.image_processor.load_image(temp_path)
                            processed = st.session_state.image_processor.preprocess_image(image)

                            prediction = st.session_state.cnn_model.predict(np.array([processed]))
                            probability = st.session_state.cnn_model.predict_proba(np.array([processed]))

                            temp_path.unlink()

                            st.markdown("###  Kết quả phân tích")

                            col1, col2 = st.columns(2)

                            with col1:
                                if prediction[0]:
                                    st.error("⚠️ **PHÁT HIỆN SỰ CỐ**")
                                else:
                                    st.success(" **BÌNH THƯỜNG**")

                            with col2:
                                st.metric("Xác suất", f"{probability[0]:.4f}")

                            st.progress(float(probability[0]))

                        except Exception as e:
                            st.error(f"❌ Lỗi: {str(e)}")
                            st.exception(e)

            st.markdown("---")
            st.markdown("### 🖼️ Test với ảnh có sẵn")

            data_path = Path("data/images")
            test_folder = st.selectbox(
                "Chọn folder để test",
                ["normal", "incident"],
                help="Chọn folder chứa ảnh để test"
            )

            test_dir = data_path / test_folder
            if test_dir.exists():
                image_files = load_image_files(test_dir)

                if image_files:
                    selected_image = st.selectbox(
                        "Chọn ảnh",
                        [str(f) for f in image_files],
                        format_func=lambda x: Path(x).name
                    )

                    if selected_image:
                        img_path = Path(selected_image)
                        img = Image.open(img_path)
                        st.image(img, caption=img_path.name, use_container_width=True)

                        if st.button(" Phân tích ảnh này", type="primary"):
                            with st.spinner("Đang xử lý..."):
                                try:
                                    image = st.session_state.image_processor.load_image(img_path)
                                    processed = st.session_state.image_processor.preprocess_image(image)

                                    prediction = st.session_state.cnn_model.predict(np.array([processed]))
                                    probability = st.session_state.cnn_model.predict_proba(np.array([processed]))

                                    st.markdown("###  Kết quả phân tích")

                                    col1, col2 = st.columns(2)

                                    with col1:
                                        if prediction[0]:
                                            st.error("⚠️ **PHÁT HIỆN SỰ CỐ**")
                                        else:
                                            st.success(" **BÌNH THƯỜNG**")

                                    with col2:
                                        st.metric("Xác suất", f"{probability[0]:.4f}")

                                    expected = "Có sự cố" if test_folder == "incident" else "Bình thường"
                                    actual = "Có sự cố" if prediction[0] else "Bình thường"

                                    if expected == actual:
                                        st.success(f" Dự đoán đúng! (Expected: {expected})")
                                    else:
                                        st.warning(f"⚠️ Dự đoán sai! (Expected: {expected}, Got: {actual})")

                                except Exception as e:
                                    st.error(f"❌ Lỗi: {str(e)}")
                                    st.exception(e)
                else:
                    st.warning(f"⚠️ Không tìm thấy ảnh trong {test_folder}")
            else:
                st.error(f"❌ Không tìm thấy folder: {test_dir}")

        except Exception as e:
            st.error(f"❌ Lỗi khi load mô hình: {str(e)}")
            st.exception(e)

elif page == " Kết quả & Metrics":
    st.title(" Kết quả & Metrics")
    st.markdown("---")

    if st.session_state.training_results:
        results = st.session_state.training_results

        st.markdown("###  Metrics huấn luyện")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Train Accuracy", f"{results.get('train_metrics', {}).get('accuracy', 0):.4f}")
        with col2:
            st.metric("Val Accuracy", f"{results.get('val_metrics', {}).get('accuracy', 0):.4f}")
        with col3:
            st.metric("Test Accuracy", f"{results.get('test_metrics', {}).get('accuracy', 0):.4f}")
        with col4:
            st.metric("Test F1-Score", f"{results.get('test_metrics', {}).get('f1_score', 0):.4f}")

        st.markdown("---")

        st.markdown("### 📋 Chi tiết metrics")

        metrics_df = pd.DataFrame({
            'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score'],
            'Train': [
                results.get('train_metrics', {}).get('accuracy', 0),
                results.get('train_metrics', {}).get('precision', 0),
                results.get('train_metrics', {}).get('recall', 0),
                results.get('train_metrics', {}).get('f1_score', 0)
            ],
            'Validation': [
                results.get('val_metrics', {}).get('accuracy', 0),
                results.get('val_metrics', {}).get('precision', 0),
                results.get('val_metrics', {}).get('recall', 0),
                results.get('val_metrics', {}).get('f1_score', 0)
            ],
            'Test': [
                results.get('test_metrics', {}).get('accuracy', 0),
                results.get('test_metrics', {}).get('precision', 0),
                results.get('test_metrics', {}).get('recall', 0),
                results.get('test_metrics', {}).get('f1_score', 0)
            ]
        })

        st.dataframe(metrics_df, use_container_width=True)

        if results.get('model_path'):
            st.info(f"💾 Mô hình đã lưu tại: {results['model_path']}")
    else:
        st.info("ℹ️ Chưa có kết quả huấn luyện. Vui lòng huấn luyện mô hình trước.")

    st.markdown("---")
    st.markdown("### ℹ️ Thông tin mô hình")

    model_path = Path("models/CNN_model")
    if model_path.exists():
        st.success(" Mô hình đã được huấn luyện")
        st.code(str(model_path.absolute()))
    else:
        st.warning("⚠️ Chưa có mô hình được huấn luyện")