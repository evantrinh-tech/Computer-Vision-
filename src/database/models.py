from sqlalchemy import (
    Column, Integer, String, Float, Boolean, DateTime, Text, JSON,
    ForeignKey, Index, CheckConstraint
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime
from typing import Optional, Dict, Any

Base = declarative_base()


class Incident(Base):
    __tablename__ = 'incidents'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    
    timestamp = Column(DateTime, nullable=False, index=True, comment='Thời điểm phát hiện sự cố')
    camera_id = Column(String(100), nullable=True, index=True, comment='ID camera phát hiện')
    location = Column(String(255), nullable=True, comment='Vị trí (GPS, địa chỉ)')
    latitude = Column(Float, nullable=True, comment='Vĩ độ (GPS)')
    longitude = Column(Float, nullable=True, comment='Kinh độ (GPS)')
    
    incident_type = Column(
        String(50), 
        nullable=True, 
        comment='Loại sự cố: accident, breakdown, congestion, event, other'
    )
    severity = Column(
        String(20),
        nullable=True,
        default='medium',
        comment='Mức độ nghiêm trọng: low, medium, high, critical'
    )
    
    confidence_score = Column(Float, nullable=False, comment='Độ tin cậy từ model (0-1)')
    model_version = Column(String(50), nullable=False, comment='Version của model đã sử dụng')
    threshold = Column(Float, nullable=False, comment='Threshold đã sử dụng để predict')
    
    rule_version = Column(String(50), nullable=True, comment='Version của temporal confirmation rules')
    confirmation_method = Column(
        String(50),
        nullable=True,
        comment='Phương pháp xác nhận: k_frames, moving_avg, manual'
    )
    
    status = Column(
        String(20),
        nullable=False,
        default='detected',
        index=True,
        comment='Trạng thái: detected, confirmed, resolved, false_alarm'
    )
    
    image_path = Column(Text, nullable=True, comment='Đường dẫn ảnh (local hoặc S3 key)')
    video_path = Column(Text, nullable=True, comment='Đường dẫn video (local hoặc S3 key)')
    media_storage_type = Column(
        String(20),
        nullable=True,
        default='local',
        comment='Loại storage: local, s3, gcs'
    )
    media_url = Column(Text, nullable=True, comment='Signed URL hoặc public URL cho media')
    
    metadata = Column(JSON, nullable=True, comment='Thông tin bổ sung (JSON)')
    
    latency_ms = Column(Float, nullable=True, comment='Thời gian xử lý (milliseconds)')
    processing_time_ms = Column(Float, nullable=True, comment='Thời gian inference (milliseconds)')
    
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow, index=True)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)
    resolved_at = Column(DateTime, nullable=True, comment='Thời điểm sự cố được giải quyết')
    
    predictions = relationship("Prediction", back_populates="incident", cascade="all, delete-orphan")
    alerts = relationship("Alert", back_populates="incident", cascade="all, delete-orphan")
    media_files = relationship("IncidentMedia", back_populates="incident", cascade="all, delete-orphan")
    
    __table_args__ = (
        Index('idx_incident_timestamp_camera', 'timestamp', 'camera_id'),
        Index('idx_incident_status_timestamp', 'status', 'timestamp'),
        Index('idx_incident_type_timestamp', 'incident_type', 'timestamp'),
    )
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None,
            'camera_id': self.camera_id,
            'location': self.location,
            'latitude': self.latitude,
            'longitude': self.longitude,
            'incident_type': self.incident_type,
            'severity': self.severity,
            'confidence_score': self.confidence_score,
            'model_version': self.model_version,
            'threshold': self.threshold,
            'rule_version': self.rule_version,
            'confirmation_method': self.confirmation_method,
            'status': self.status,
            'image_path': self.image_path,
            'video_path': self.video_path,
            'media_storage_type': self.media_storage_type,
            'media_url': self.media_url,
            'metadata': self.metadata,
            'latency_ms': self.latency_ms,
            'processing_time_ms': self.processing_time_ms,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
            'resolved_at': self.resolved_at.isoformat() if self.resolved_at else None,
        }


class Prediction(Base):
    __tablename__ = 'predictions'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    
    incident_id = Column(
        Integer,
        ForeignKey('incidents.id', ondelete='CASCADE'),
        nullable=True,
        index=True,
        comment='ID của incident (nếu đã được confirm)'
    )
    
    model_name = Column(String(50), nullable=False, index=True, comment='Tên model: CNN, ANN, RNN, Hybrid')
    model_version = Column(String(50), nullable=False, comment='Version của model')
    
    prediction = Column(Boolean, nullable=False, comment='Dự đoán: True = có sự cố, False = không')
    probability = Column(Float, nullable=False, comment='Xác suất (0-1)')
    threshold = Column(Float, nullable=False, comment='Threshold đã sử dụng')
    
    camera_id = Column(String(100), nullable=True, index=True, comment='ID camera')
    frame_number = Column(Integer, nullable=True, comment='Số frame (nếu là video)')
    timestamp = Column(DateTime, nullable=False, index=True, comment='Thời điểm prediction')
    
    processing_time_ms = Column(Float, nullable=True, comment='Thời gian inference (ms)')
    latency_ms = Column(Float, nullable=True, comment='Tổng thời gian xử lý (ms)')
    
    ground_truth = Column(Boolean, nullable=True, comment='Nhãn thực tế (để evaluation)')
    is_correct = Column(Boolean, nullable=True, comment='Prediction có đúng không')
    
    metadata = Column(JSON, nullable=True, comment='Thông tin bổ sung')
    
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow, index=True)
    
    incident = relationship("Incident", back_populates="predictions")
    
    __table_args__ = (
        Index('idx_prediction_timestamp_camera', 'timestamp', 'camera_id'),
        Index('idx_prediction_model_timestamp', 'model_name', 'timestamp'),
    )


class ModelRun(Base):
    __tablename__ = 'model_runs'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    
    mlflow_run_id = Column(String(100), nullable=True, unique=True, index=True, comment='MLflow run ID')
    experiment_name = Column(String(100), nullable=True, index=True, comment='Tên experiment')
    
    model_name = Column(String(50), nullable=False, comment='Tên model')
    model_version = Column(String(50), nullable=False, comment='Version')
    
    training_config = Column(JSON, nullable=True, comment='Cấu hình training (JSON)')
    hyperparameters = Column(JSON, nullable=True, comment='Hyperparameters (JSON)')
    
    train_metrics = Column(JSON, nullable=True, comment='Metrics trên train set (JSON)')
    val_metrics = Column(JSON, nullable=True, comment='Metrics trên validation set (JSON)')
    test_metrics = Column(JSON, nullable=True, comment='Metrics trên test set (JSON)')
    
    n_train_samples = Column(Integer, nullable=True, comment='Số samples train')
    n_val_samples = Column(Integer, nullable=True, comment='Số samples validation')
    n_test_samples = Column(Integer, nullable=True, comment='Số samples test')
    
    model_path = Column(Text, nullable=True, comment='Đường dẫn model file')
    artifacts_path = Column(Text, nullable=True, comment='Đường dẫn artifacts (plots, etc.)')
    
    status = Column(
        String(20),
        nullable=False,
        default='running',
        comment='Trạng thái: running, completed, failed'
    )
    
    started_at = Column(DateTime, nullable=False, default=datetime.utcnow, index=True)
    completed_at = Column(DateTime, nullable=True, comment='Thời điểm hoàn thành')
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    
    __table_args__ = (
        Index('idx_model_run_experiment_started', 'experiment_name', 'started_at'),
    )


class Alert(Base):
    __tablename__ = 'alerts'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    
    incident_id = Column(
        Integer,
        ForeignKey('incidents.id', ondelete='CASCADE'),
        nullable=False,
        index=True,
        comment='ID của incident'
    )
    
    alert_type = Column(
        String(50),
        nullable=False,
        comment='Loại alert: email, sms, push, webhook, dashboard'
    )
    recipient = Column(String(255), nullable=True, comment='Người nhận (email, phone, user_id)')
    
    title = Column(String(255), nullable=True, comment='Tiêu đề alert')
    message = Column(Text, nullable=True, comment='Nội dung alert')
    
    status = Column(
        String(20),
        nullable=False,
        default='pending',
        index=True,
        comment='Trạng thái: pending, sent, failed, read'
    )
    
    sent_at = Column(DateTime, nullable=True, comment='Thời điểm gửi')
    read_at = Column(DateTime, nullable=True, comment='Thời điểm đọc (nếu có)')
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow, index=True)
    
    metadata = Column(JSON, nullable=True, comment='Thông tin bổ sung')
    
    incident = relationship("Incident", back_populates="alerts")
    
    __table_args__ = (
        Index('idx_alert_status_created', 'status', 'created_at'),
    )


class IncidentMedia(Base):
    __tablename__ = 'incident_media'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    
    incident_id = Column(
        Integer,
        ForeignKey('incidents.id', ondelete='CASCADE'),
        nullable=False,
        index=True,
        comment='ID của incident'
    )
    
    media_type = Column(
        String(20),
        nullable=False,
        comment='Loại media: image, video, thumbnail'
    )
    file_path = Column(Text, nullable=False, comment='Đường dẫn file (local hoặc S3 key)')
    file_size_bytes = Column(Integer, nullable=True, comment='Kích thước file (bytes)')
    mime_type = Column(String(100), nullable=True, comment='MIME type: image/jpeg, video/mp4, ...')
    
    storage_type = Column(
        String(20),
        nullable=False,
        default='local',
        comment='Loại storage: local, s3, gcs'
    )
    storage_bucket = Column(String(255), nullable=True, comment='Bucket name (nếu là S3/GCS)')
    storage_key = Column(String(500), nullable=True, comment='Object key (nếu là S3/GCS)')
    
    signed_url = Column(Text, nullable=True, comment='Signed URL (temporary, có expiry)')
    public_url = Column(Text, nullable=True, comment='Public URL (nếu có)')
    
    width = Column(Integer, nullable=True, comment='Chiều rộng (pixels, cho ảnh)')
    height = Column(Integer, nullable=True, comment='Chiều cao (pixels, cho ảnh)')
    duration_seconds = Column(Float, nullable=True, comment='Độ dài (giây, cho video)')
    metadata = Column(JSON, nullable=True, comment='Thông tin bổ sung')
    
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow, index=True)
    
    incident = relationship("Incident", back_populates="media_files")
    
    __table_args__ = (
        Index('idx_media_incident_type', 'incident_id', 'media_type'),
    )

