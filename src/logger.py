"""
Logging configuration module for the Loan Approval Prediction system.
"""

import logging
import logging.handlers
from src.config import LOG_FILE, LOG_LEVEL, LOGS_DIR


def setup_logger(name: str) -> logging.Logger:
    """
    Configure and return a logger instance.

    Parameters
    ----------
    name : str
        Name of the logger (typically __name__)

    Returns
    -------
    logging.Logger
        Configured logger instance
    """
    LOGS_DIR.mkdir(exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, LOG_LEVEL))

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, LOG_LEVEL))

    # File handler
    file_handler = logging.handlers.RotatingFileHandler(
        LOG_FILE, maxBytes=5 * 1024 * 1024, backupCount=5
    )
    file_handler.setLevel(getattr(logging, LOG_LEVEL))

    # Formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    # Add handlers if not already present
    if not logger.handlers:
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)

    return logger


# Create default logger
logger = setup_logger(__name__)
