import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional, Iterator
import os
from PIL import Image

from src.utils.logger import logger

class ImageProcessor:

    def __init__(self, image_size: Tuple[int, int] = (224, 224)):

        self.image_size = image_size

    def load_image(self, image_path: Path) -> np.ndarray:

        if not image_path.exists():
            raise FileNotFoundError(f"Không tìm thấy file ảnh: {image_path}")

        if image_path.suffix.lower() == '.gif':
            try:
                pil_image = Image.open(str(image_path))
                if pil_image.mode != 'RGB':
                    pil_image = pil_image.convert('RGB')
                image_array = np.array(pil_image)
                image = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
            except Exception as e:
                raise ValueError(f"Không thể đọc file GIF {image_path}: {e}")
        else:
            image = cv2.imread(str(image_path))
            if image is None:
                raise ValueError(f"Không thể đọc ảnh: {image_path}")

        return image

    def preprocess_image(self, image: np.ndarray) -> np.ndarray:

        resized = cv2.resize(image, self.image_size)

        rgb_image = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

        normalized = rgb_image.astype(np.float32) / 255.0

        return normalized

    def detect_objects(self, image: np.ndarray) -> List[dict]:

        return []

    def extract_roi(self, image: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:

        x, y, w, h = bbox
        roi = image[y:y+h, x:x+w]
        return roi

class VideoProcessor:

    def __init__(
        self,
        image_size: Tuple[int, int] = (224, 224),
        frame_skip: int = 1
    ):

        self.image_size = image_size
        self.frame_skip = frame_skip
        self.image_processor = ImageProcessor(image_size)

    def load_video(self, video_path: Path) -> cv2.VideoCapture:

        if not video_path.exists():
            raise FileNotFoundError(f"Không tìm thấy file video: {video_path}")

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Không thể mở video: {video_path}")

        return cap

    def extract_frames(
        self,
        video_path: Path,
        output_dir: Optional[Path] = None,
        max_frames: Optional[int] = None
    ) -> List[np.ndarray]:

        cap = self.load_video(video_path)
        frames = []
        frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % self.frame_skip != 0:
                frame_count += 1
                continue

            processed_frame = self.image_processor.preprocess_image(frame)
            frames.append(processed_frame)

            if output_dir:
                output_dir.mkdir(parents=True, exist_ok=True)
                frame_path = output_dir / f"frame_{frame_count:06d}.jpg"
                cv2.imwrite(str(frame_path), frame)

            frame_count += 1

            if max_frames and len(frames) >= max_frames:
                break

        cap.release()
        logger.info(f"Đã trích xuất {len(frames)} frames từ video")

        return frames

    def process_video_stream(
        self,
        video_path: Path,
        model,
        batch_size: int = 32
    ) -> Iterator[dict]:

        cap = self.load_video(video_path)
        frame_batch = []
        frame_numbers = []
        frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                if frame_batch:
                    predictions = model.predict(np.array(frame_batch))
                    for i, (fn, pred, prob) in enumerate(zip(
                        frame_numbers,
                        predictions,
                        model.predict_proba(np.array(frame_batch))
                    )):
                        yield {
                            'frame_number': fn,
                            'has_incident': bool(pred),
                            'probability': float(prob)
                        }
                break

            processed_frame = self.image_processor.preprocess_image(frame)
            frame_batch.append(processed_frame)
            frame_numbers.append(frame_count)

            if len(frame_batch) >= batch_size:
                predictions = model.predict(np.array(frame_batch))
                probabilities = model.predict_proba(np.array(frame_batch))

                for fn, pred, prob in zip(frame_numbers, predictions, probabilities):
                    yield {
                        'frame_number': fn,
                        'has_incident': bool(pred),
                        'probability': float(prob)
                    }

                frame_batch = []
                frame_numbers = []

            frame_count += 1

        cap.release()

    def detect_incidents_in_video(
        self,
        video_path: Path,
        model,
        threshold: float = 0.5
    ) -> List[dict]:

        incidents = []
        cap = self.load_video(video_path)

        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = 0

        frame_batch = []
        frame_numbers = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            processed_frame = self.image_processor.preprocess_image(frame)
            frame_batch.append(processed_frame)
            frame_numbers.append(frame_count)

            if len(frame_batch) >= 32:
                probabilities = model.predict_proba(np.array(frame_batch))

                for fn, prob in zip(frame_numbers, probabilities):
                    if prob > threshold:
                        timestamp = fn / fps if fps > 0 else fn
                        incidents.append({
                            'frame_number': int(fn),
                            'timestamp_seconds': float(timestamp),
                            'probability': float(prob)
                        })

                frame_batch = []
                frame_numbers = []

            frame_count += 1

        if frame_batch:
            probabilities = model.predict_proba(np.array(frame_batch))
            for fn, prob in zip(frame_numbers, probabilities):
                if prob > threshold:
                    timestamp = fn / fps if fps > 0 else fn
                    incidents.append({
                        'frame_number': int(fn),
                        'timestamp_seconds': float(timestamp),
                        'probability': float(prob)
                    })

        cap.release()

        logger.info(f"Phát hiện {len(incidents)} sự cố trong video")
        return incidents