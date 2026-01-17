import os
from pathlib import Path
from typing import Optional
from pydantic_settings import BaseSettings
from pydantic import Field

class Settings(BaseSettings):
    """
    Settings: Quản lý toàn bộ cấu hình của hệ thống thông qua biến môi trường (.env).
    
    Quy tắc:
    - Ưu tiên đọc từ Environment Variables.
    - Nếu không có, dùng giá trị default.
    - Hỗ trợ MLflow, Database (PostgreSQL), Kafka và API configs.
    """

    # 1. Cấu hình MLflow (Tracking Server)
    mlflow_tracking_uri: str = Field(
        default="http://localhost:5000",
        env="MLFLOW_TRACKING_URI"
    )
    mlflow_experiment_name: str = Field(
        default="traffic-incident-detection",
        env="MLFLOW_EXPERIMENT_NAME"
    )

    # 2. Cấu hình Database (PostgreSQL)
    database_url: str = Field(
        default="postgresql://user:password@localhost:5432/traffic_db",
        env="DATABASE_URL"
    )

    # 3. Cấu hình Kafka (Message Queue)
    kafka_bootstrap_servers: str = Field(
        default="localhost:9092",
        env="KAFKA_BOOTSTRAP_SERVERS"
    )
    kafka_topic_sensor_data: str = Field(
        default="sensor-data",
        env="KAFKA_TOPIC_SENSOR_DATA"
    )
    kafka_topic_camera_data: str = Field(
        default="camera-data",
        env="KAFKA_TOPIC_CAMERA_DATA"
    )

    api_host: str = Field(default="0.0.0.0", env="API_HOST")
    api_port: int = Field(default=8000, env="API_PORT")
    api_workers: int = Field(default=4, env="API_WORKERS")

    # 4. Cấu hình Model Registry và Data Paths
    model_registry_path: Path = Field(
        default=Path("./models"),
        env="MODEL_REGISTRY_PATH"
    )
    default_model_version: str = Field(
        default="latest",
        env="DEFAULT_MODEL_VERSION"
    )

    data_raw_path: Path = Field(
        default=Path("./data/raw"),
        env="DATA_RAW_PATH"
    )
    data_processed_path: Path = Field(
        default=Path("./data/processed"),
        env="DATA_PROCESSED_PATH"
    )

    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    log_file: Optional[Path] = Field(
        default=Path("./logs/app.log"),
        env="LOG_FILE"
    )

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False

settings = Settings()

def get_settings() -> Settings:

    return settings