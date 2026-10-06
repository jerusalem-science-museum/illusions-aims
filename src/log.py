import logging
import os
from logging.handlers import RotatingFileHandler

import constant as cfg


def get_logger() -> logging.Logger:
    """App logger: rotating files in LOG_FOLDER (log.txt, log.txt.1, ...) plus the console."""
    logger = logging.getLogger("aims")
    if not logger.handlers:
        os.makedirs(cfg.LOG_FOLDER, exist_ok=True)
        fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s", "%Y-%m-%d %H:%M:%S")
        file_handler = RotatingFileHandler(
            os.path.join(cfg.LOG_FOLDER, "log.txt"),
            maxBytes=cfg.MAX_SIZE_PER_LOG_FILE,
            backupCount=max(0, cfg.BACKUP_COUNT - 1),
            encoding="utf-8",
        )
        console = logging.StreamHandler()
        for h in (file_handler, console):
            h.setFormatter(fmt)
            logger.addHandler(h)
        logger.setLevel(logging.INFO)
    return logger
