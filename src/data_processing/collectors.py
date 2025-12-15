import json
import time
from typing import Dict, List, Optional, Iterator
from datetime import datetime
from kafka import KafkaConsumer, KafkaProducer
from kafka.errors import KafkaError
import pandas as pd

from src.utils.logger import logger
from src.utils.config import settings

class SensorDataCollector:

    def __init__(
        self,
        bootstrap_servers: Optional[str] = None,
        topic: Optional[str] = None
    ):

        self.bootstrap_servers = bootstrap_servers or settings.kafka_bootstrap_servers
        self.topic = topic or settings.kafka_topic_sensor_data
        self.consumer: Optional[KafkaConsumer] = None

    def connect(self) -> None:

        try:
            self.consumer = KafkaConsumer(
                self.topic,
                bootstrap_servers=self.bootstrap_servers,
                value_deserializer=lambda m: json.loads(m.decode('utf-8')),
                consumer_timeout_ms=1000,
                auto_offset_reset='latest',
                enable_auto_commit=True
            )
            logger.info(f"Đã kết nối đến Kafka topic: {self.topic}")
        except KafkaError as e:
            logger.error(f"Lỗi kết nối Kafka: {e}")
            raise

    def collect(self, timeout: Optional[float] = None) -> Iterator[Dict]:

        if self.consumer is None:
            self.connect()

        start_time = time.time()

        for message in self.consumer:
            if timeout and (time.time() - start_time) > timeout:
                break

            data = message.value
            data['timestamp'] = datetime.now().isoformat()
            data['source'] = 'sensor'

            yield data

    def collect_batch(self, batch_size: int = 100) -> List[Dict]:

        batch = []
        for data in self.collect():
            batch.append(data)
            if len(batch) >= batch_size:
                break
        return batch

    def close(self) -> None:

        if self.consumer:
            self.consumer.close()
            logger.info("Đã đóng kết nối Kafka")

class CameraDataCollector:

    def __init__(
        self,
        bootstrap_servers: Optional[str] = None,
        topic: Optional[str] = None
    ):
        self.bootstrap_servers = bootstrap_servers or settings.kafka_bootstrap_servers
        self.topic = topic or settings.kafka_topic_camera_data
        self.consumer: Optional[KafkaConsumer] = None

    def connect(self) -> None:

        try:
            self.consumer = KafkaConsumer(
                self.topic,
                bootstrap_servers=self.bootstrap_servers,
                value_deserializer=lambda m: json.loads(m.decode('utf-8')),
                consumer_timeout_ms=1000,
                auto_offset_reset='latest',
                enable_auto_commit=True
            )
            logger.info(f"Đã kết nối đến Kafka topic: {self.topic}")
        except KafkaError as e:
            logger.error(f"Lỗi kết nối Kafka: {e}")
            raise

    def collect(self, timeout: Optional[float] = None) -> Iterator[Dict]:

        if self.consumer is None:
            self.connect()

        start_time = time.time()

        for message in self.consumer:
            if timeout and (time.time() - start_time) > timeout:
                break

            data = message.value
            data['timestamp'] = datetime.now().isoformat()
            data['source'] = 'camera'

            yield data

    def close(self) -> None:

        if self.consumer:
            self.consumer.close()

class SimulatedDataCollector:

    def __init__(self, seed: int = 42):

        import numpy as np
        self.np = np
        self.np.random.seed(seed)

    def generate_sensor_data(
        self,
        n_samples: int = 100,
        has_incident: bool = False
    ) -> pd.DataFrame:

        data = []

        for i in range(n_samples):
            if not has_incident or i < n_samples * 0.7:
                volume = self.np.random.normal(1000, 200)
                speed = self.np.random.normal(60, 10)
                occupancy = self.np.random.normal(0.3, 0.1)
            else:
                volume = self.np.random.normal(300, 100)
                speed = self.np.random.normal(20, 5)
                occupancy = self.np.random.normal(0.8, 0.1)

            data.append({
                'timestamp': datetime.now().isoformat(),
                'detector_id': f'detector_{i % 10}',
                'volume': max(0, volume),
                'speed': max(0, speed),
                'occupancy': max(0, min(1, occupancy)),
                'has_incident': 1 if (has_incident and i >= n_samples * 0.7) else 0
            })

        return pd.DataFrame(data)