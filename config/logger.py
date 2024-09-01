import logging
import os
from datetime import datetime
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path


FORMAT = (
    "%(asctime)s - %(levelname)-7s - %(filename)-20s"
    +" - %(lineno)-4s - %(message)s"
)

def setup_logging():
    """Setup logging"""
    LOG_LEVEL = os.getenv("LEVEL", "INFO").upper()
    LOG_DIR_PATH = Path("/app/data/logs/")
    LOG_FILE_PATH = LOG_DIR_PATH / (
        f"{datetime.now().strftime('%Y%m%d')}.log")
    LOG_DIR_PATH.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=getattr(logging, LOG_LEVEL, logging.INFO),
        format=FORMAT,
        datefmt="%Y-%m-%d %H:%M:%S",
        filename=LOG_FILE_PATH
    )
    logger = logging.getLogger(__name__)

    return logger
