from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import numpy as np
import pandas as pd
from pathlib import Path
import uvicorn

from src.serving.predictor import ModelPredictor
from src.serving.monitoring import MetricsCollector
from src.utils.config import settings
from src.utils.logger import logger

app = FastAPI(
    title="Traffic Incident Detection API",
    description="API để phát hiện sự cố giao thông sử dụng Neural Network (CNN, RNN, RBFNN...)",
    version="0.1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

predictor = ModelPredictor()
metrics_collector = MetricsCollector()

class SensorData(BaseModel):
    """
    Schema dữ liệu cảm biến đầu vào.
    """
    timestamp: str = Field(..., description="Thời gian ghi nhận (ISO format)")
    detector_id: str = Field(..., description="ID của cảm biến/camera")
    volume: float = Field(..., ge=0, description="Lưu lượng xe (vehicles/hour)")
    speed: float = Field(..., ge=0, le=200, description="Tốc độ trung bình (km/h)")
    occupancy: float = Field(..., ge=0, le=1, description="Độ chiếm dụng mặt đường (0.0 - 1.0)")

class PredictionRequest(BaseModel):

    data: List[SensorData]

class PredictionResponse(BaseModel):

    predictions: List[Dict[str, Any]]
    model_version: str
    processing_time: float

class HealthResponse(BaseModel):

    status: str
    model_loaded: bool
    model_version: Optional[str] = None

@app.get("/", tags=["General"])
async def root():
    """
    Trang chủ API.
    """
    return {
        "message": "Traffic Incident Detection API",
        "version": "0.1.0",
        "docs": "/docs"
    }

@app.get("/health", response_model=HealthResponse, tags=["General"])
async def health_check():

    model_loaded = predictor.is_model_loaded()
    return HealthResponse(
        status="healthy" if model_loaded else "model_not_loaded",
        model_loaded=model_loaded,
        model_version=predictor.get_model_version() if model_loaded else None
    )

@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(
    request: PredictionRequest,
    background_tasks: BackgroundTasks
):
    """
    Endpoint nhận dữ liệu giao thông và trả về kết quả dự đoán sự cố.
    - Xử lý Batch prediction (nhiều mẫu cùng lúc).
    - Log metrics xuống background task để không chặn request.
    """

    import time
    start_time = time.time()

    try:
        data_dicts = [item.dict() for item in request.data]
        df = pd.DataFrame(data_dicts)

        predictions = predictor.predict(df)

        results = []
        for i, pred in enumerate(predictions):
            results.append({
                "sample_index": i,
                "has_incident": bool(pred['prediction']),
                "probability": float(pred['probability']),
                "timestamp": data_dicts[i]['timestamp'],
                "detector_id": data_dicts[i]['detector_id']
            })

        processing_time = time.time() - start_time

        background_tasks.add_task(
            metrics_collector.log_prediction,
            len(predictions),
            processing_time
        )

        return PredictionResponse(
            predictions=results,
            model_version=predictor.get_model_version(),
            processing_time=processing_time
        )

    except Exception as e:
        logger.error(f"Error in prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/metrics", tags=["Monitoring"])
async def get_metrics():

    return metrics_collector.get_metrics()

@app.post("/model/reload", tags=["Model Management"])
async def reload_model(model_path: Optional[str] = None):

    try:
        if model_path:
            predictor.load_model(Path(model_path))
        else:
            predictor.load_default_model()

        return {"status": "success", "message": "Model reloaded successfully"}
    except Exception as e:
        logger.error(f"Error reloading model: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/model/info", tags=["Model Management"])
async def get_model_info():

    if not predictor.is_model_loaded():
        raise HTTPException(status_code=404, detail="Model not loaded")

    return {
        "model_type": predictor.model_type,
        "model_version": predictor.get_model_version(),
        "model_info": predictor.get_model_info()
    }

def main():

    try:
        predictor.load_default_model()
        logger.info("Default model loaded successfully")
    except Exception as e:
        logger.warning(f"Could not load default model: {e}")

    uvicorn.run(
        "src.serving.api:app",
        host=settings.api_host,
        port=settings.api_port,
        workers=settings.api_workers,
        reload=False
    )

if __name__ == '__main__':
    main()