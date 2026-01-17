import numpy as np
from typing import List, Dict, Optional, Tuple
from enum import Enum
from dataclasses import dataclass
from datetime import datetime, timedelta
import time

from src.utils.logger import logger


class IncidentStatus(Enum):
    DETECTED = "detected"
    CONFIRMED = "confirmed"
    FALSE_ALARM = "false_alarm"
    RESOLVED = "resolved"


@dataclass
class IncidentEvent:
    status: IncidentStatus
    start_frame: int
    end_frame: Optional[int] = None
    start_timestamp: float = 0.0
    end_timestamp: Optional[float] = None
    max_probability: float = 0.0
    avg_probability: float = 0.0
    confirmation_method: str = ""
    metadata: Dict = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class TemporalConfirmation:
    """
    TemporalConfirmation: Thuật toán lọc nhiễu và xác nhận sự cố theo thời gian.
    
    Vấn đề: Model AI có thể báo động giả (False Alarm) do nhiễu trong 1-2 frame.
    Giải pháp: Chỉ xác nhận là "Sự cố" nếu sự cố tồn tại (persist) qua nhiều frame liên tiếp.
    
    Phương pháp:
    1. K-Consecutive Frames: K frame liên tiếp đều vượt ngưỡng threshold.
    2. Moving Average: Trung bình xác suất trong cửa sổ window_size vượt ngưỡng.
    3. Cooldown: Sau khi xác nhận sự cố, không báo lại trong một khoảng thời gian (tránh spam alert).
    """

    def __init__(
        self,
        k_frames: int = 5,
        window_size: int = 10,
        threshold: float = 0.5,
        cooldown_seconds: float = 30.0,
        fps: float = 30.0,
        use_k_frames: bool = True,
        use_moving_avg: bool = True,
        moving_avg_threshold: float = 0.4
    ):
        """
        Args:
            k_frames: Số lượng frame liên tiếp cần để confirm.
            window_size: Kích thước cửa sổ trượt (cho moving average).
            threshold: Ngưỡng xác suất để coi là detection.
            cooldown_seconds: Thời gian nghỉ giữa các lần cảnh báo.
        """
        self.k_frames = k_frames
        self.window_size = window_size
        self.threshold = threshold
        self.cooldown_seconds = cooldown_seconds
        self.fps = fps
        self.use_k_frames = use_k_frames
        self.use_moving_avg = use_moving_avg
        self.moving_avg_threshold = moving_avg_threshold
        
        self.probability_buffer: List[float] = []
        self.frame_buffer: List[int] = []
        self.timestamp_buffer: List[float] = []
        self.current_event: Optional[IncidentEvent] = None
        self.last_confirmed_time: Optional[float] = None
        self.confirmed_events: List[IncidentEvent] = []
        
        logger.info(
            f"TemporalConfirmation initialized: "
            f"K={k_frames}, window={window_size}, threshold={threshold}, "
            f"cooldown={cooldown_seconds}s"
        )

    def process_frame(
        self,
        frame_number: int,
        probability: float,
        timestamp: Optional[float] = None
    ) -> Optional[IncidentEvent]:
        """
        Xử lý từng frame video hoặc từng mẫu dữ liệu time-series.
        Returns:
            IncidentEvent nếu sự cố được xác nhận (Confirmed), ngược lại None.
        """
        if timestamp is None:
            timestamp = frame_number / self.fps if self.fps > 0 else frame_number
        
        self.probability_buffer.append(probability)
        self.frame_buffer.append(frame_number)
        self.timestamp_buffer.append(timestamp)
        
        if len(self.probability_buffer) > self.window_size * 2:
            self.probability_buffer = self.probability_buffer[-self.window_size * 2:]
            self.frame_buffer = self.frame_buffer[-self.window_size * 2:]
            self.timestamp_buffer = self.timestamp_buffer[-self.window_size * 2:]
        
        if self.last_confirmed_time is not None:
            time_since_last = timestamp - self.last_confirmed_time
            if time_since_last < self.cooldown_seconds:
                return None
        
        detected = probability > self.threshold
        
        if detected:
            if self.current_event is None:
                self.current_event = IncidentEvent(
                    status=IncidentStatus.DETECTED,
                    start_frame=frame_number,
                    start_timestamp=timestamp,
                    max_probability=probability,
                    avg_probability=probability,
                    metadata={"probabilities": [probability]}
                )
            else:
                self.current_event.max_probability = max(
                    self.current_event.max_probability, probability
                )
                self.current_event.metadata["probabilities"].append(probability)
                probs = self.current_event.metadata["probabilities"]
                self.current_event.avg_probability = np.mean(probs)
        else:
            if self.current_event is not None:
                self.current_event.end_frame = frame_number
                self.current_event.end_timestamp = timestamp
                self.current_event = None
        
        if self.current_event is not None:
            confirmed = self._check_confirmation()
            if confirmed:
                self.current_event.status = IncidentStatus.CONFIRMED
                self.current_event.end_frame = frame_number
                self.current_event.end_timestamp = timestamp
                self.last_confirmed_time = timestamp
                
                confirmed_event = self.current_event
                self.confirmed_events.append(confirmed_event)
                self.current_event = None
                
                logger.info(
                    f"Incident confirmed at frame {frame_number} "
                    f"(method: {confirmed_event.confirmation_method}, "
                    f"prob: {confirmed_event.avg_probability:.3f})"
                )
                
                return confirmed_event
        
        return None

    def _check_confirmation(self) -> bool:
        if self.current_event is None:
            return False
        
        if self.use_k_frames:
            if self._check_k_frames():
                self.current_event.confirmation_method = "k_frames"
                return True
        
        if self.use_moving_avg:
            if self._check_moving_avg():
                self.current_event.confirmation_method = "moving_avg"
                return True
        
        return False

    def _check_k_frames(self) -> bool:
        if len(self.probability_buffer) < self.k_frames:
            return False
        
        recent_probs = self.probability_buffer[-self.k_frames:]
        all_above_threshold = all(p > self.threshold for p in recent_probs)
        
        return all_above_threshold

    def _check_moving_avg(self) -> bool:
        if len(self.probability_buffer) < self.window_size:
            return False
        
        recent_probs = self.probability_buffer[-self.window_size:]
        avg_prob = np.mean(recent_probs)
        
        return avg_prob > self.moving_avg_threshold

    def process_stream(
        self,
        probabilities: List[float],
        timestamps: Optional[List[float]] = None,
        start_frame: int = 0
    ) -> List[IncidentEvent]:
        confirmed_events = []
        
        for i, prob in enumerate(probabilities):
            frame_num = start_frame + i
            timestamp = timestamps[i] if timestamps else None
            
            event = self.process_frame(frame_num, prob, timestamp)
            if event is not None:
                confirmed_events.append(event)
        
        return confirmed_events

    def mark_false_alarm(self, event: IncidentEvent) -> None:
        event.status = IncidentStatus.FALSE_ALARM
        event.confirmation_method = "manual_review"
        logger.info(f"Event at frame {event.start_frame} marked as false alarm")

    def mark_resolved(self, event: IncidentEvent) -> None:
        event.status = IncidentStatus.RESOLVED
        logger.info(f"Event at frame {event.start_frame} marked as resolved")

    def get_statistics(self) -> Dict[str, float]:
        if not self.confirmed_events:
            return {
                "total_events": 0,
                "confirmed": 0,
                "false_alarms": 0,
                "resolved": 0,
                "avg_duration": 0.0,
                "avg_probability": 0.0
            }
        
        confirmed = sum(1 for e in self.confirmed_events 
                       if e.status == IncidentStatus.CONFIRMED)
        false_alarms = sum(1 for e in self.confirmed_events 
                          if e.status == IncidentStatus.FALSE_ALARM)
        resolved = sum(1 for e in self.confirmed_events 
                      if e.status == IncidentStatus.RESOLVED)
        
        durations = []
        probs = []
        for event in self.confirmed_events:
            if event.end_timestamp is not None:
                durations.append(event.end_timestamp - event.start_timestamp)
            probs.append(event.avg_probability)
        
        return {
            "total_events": len(self.confirmed_events),
            "confirmed": confirmed,
            "false_alarms": false_alarms,
            "resolved": resolved,
            "avg_duration": np.mean(durations) if durations else 0.0,
            "avg_probability": np.mean(probs) if probs else 0.0
        }

    def reset(self) -> None:
        self.probability_buffer = []
        self.frame_buffer = []
        self.timestamp_buffer = []
        self.current_event = None
        self.last_confirmed_time = None
        self.confirmed_events = []
        logger.info("TemporalConfirmation state reset")


def tune_temporal_params(
    probabilities: List[float],
    ground_truth: List[int],
    fps: float = 30.0,
    k_range: Tuple[int, int] = (3, 10),
    window_range: Tuple[int, int] = (5, 20),
    threshold_range: Tuple[float, float] = (0.3, 0.7),
    cooldown_range: Tuple[float, float] = (10.0, 60.0)
) -> Dict[str, float]:
    from sklearn.metrics import recall_score, precision_score, f1_score
    
    best_params = None
    best_f1 = 0.0
    best_recall = 0.0
    best_precision = 0.0
    
    for k in range(k_range[0], k_range[1] + 1, 2):
        for window in range(window_range[0], window_range[1] + 1, 5):
                for threshold in np.arange(threshold_range[0], threshold_range[1], 0.1):
                for cooldown in np.arange(cooldown_range[0], cooldown_range[1], 10.0):
                    confirmer = TemporalConfirmation(
                        k_frames=k,
                        window_size=window,
                        threshold=threshold,
                        cooldown_seconds=cooldown,
                        fps=fps
                    )
                    
                    events = confirmer.process_stream(probabilities)
                    
                    predictions = [0] * len(probabilities)
                    for event in events:
                        start_idx = event.start_frame
                        end_idx = event.end_frame if event.end_frame else len(probabilities)
                        for i in range(start_idx, min(end_idx, len(predictions))):
                            predictions[i] = 1
                    
                    recall = recall_score(ground_truth, predictions, zero_division=0)
                    precision = precision_score(ground_truth, predictions, zero_division=0)
                    f1 = f1_score(ground_truth, predictions, zero_division=0)
                    
                    if recall >= 0.9 and f1 > best_f1:
                        best_f1 = f1
                        best_recall = recall
                        best_precision = precision
                        best_params = {
                            "k_frames": k,
                            "window_size": window,
                            "threshold": threshold,
                            "cooldown_seconds": cooldown,
                            "f1_score": f1,
                            "recall": recall,
                            "precision": precision
                        }
    
    if best_params is None:
        logger.warning("Không tìm thấy params tốt, sử dụng default")
        return {
            "k_frames": 5,
            "window_size": 10,
            "threshold": 0.5,
            "cooldown_seconds": 30.0
        }
    
    logger.info(f"Best params found: {best_params}")
    return best_params

