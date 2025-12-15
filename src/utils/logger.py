import logging
import sys
from pathlib import Path
from typing import Optional
from pythonjsonlogger import jsonlogger

from src.utils.config import settings

def setup_logger(
    name: str = __name__,
    log_level: Optional[str] = None,
    log_file: Optional[Path] = None,
    json_format: bool = False
) -> logging.Logger:

    logger = logging.getLogger(name)

    level = log_level or settings.log_level
    logger.setLevel(getattr(logging, level.upper()))

    logger.handlers.clear()

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, level.upper()))

    if json_format:
        formatter = jsonlogger.JsonFormatter(
            "%(asctime)s %(name)s %(levelname)s %(message)s"
        )
    else:
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )

    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if log_file:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(getattr(logging, level.upper()))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger

logger = setup_logger("traffic_incident_detection")