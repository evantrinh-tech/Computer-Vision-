from typing import Dict, Any
from collections import deque
import time
from threading import Lock

from src.utils.logger import logger

class MetricsCollector:

    def __init__(self, max_history: int = 1000):

        self.max_history = max_history
        self.lock = Lock()

        self.total_predictions = 0
        self.total_incidents_detected = 0
        self.total_processing_time = 0.0

        self.processing_times = deque(maxlen=max_history)
        self.prediction_counts = deque(maxlen=max_history)

    def log_prediction(
        self,
        n_predictions: int,
        processing_time: float,
        n_incidents: int = 0
    ) -> None:

        with self.lock:
            self.total_predictions += n_predictions
            self.total_incidents_detected += n_incidents
            self.total_processing_time += processing_time

            self.processing_times.append(processing_time)
            self.prediction_counts.append(n_predictions)

    def get_metrics(self) -> Dict[str, Any]:

        with self.lock:
            avg_processing_time = (
                sum(self.processing_times) / len(self.processing_times)
                if self.processing_times else 0.0
            )

            avg_predictions_per_batch = (
                sum(self.prediction_counts) / len(self.prediction_counts)
                if self.prediction_counts else 0.0
            )

            throughput = (
                avg_predictions_per_batch / avg_processing_time
                if avg_processing_time > 0 else 0.0
            )

            return {
                'total_predictions': self.total_predictions,
                'total_incidents_detected': self.total_incidents_detected,
                'total_processing_time_seconds': self.total_processing_time,
                'average_processing_time_seconds': avg_processing_time,
                'average_predictions_per_batch': avg_predictions_per_batch,
                'throughput_predictions_per_second': throughput,
                'incident_detection_rate': (
                    self.total_incidents_detected / self.total_predictions
                    if self.total_predictions > 0 else 0.0
                )
            }

    def reset(self) -> None:

        with self.lock:
            self.total_predictions = 0
            self.total_incidents_detected = 0
            self.total_processing_time = 0.0
            self.processing_times.clear()
            self.prediction_counts.clear()

        logger.info("Metrics đã được reset")